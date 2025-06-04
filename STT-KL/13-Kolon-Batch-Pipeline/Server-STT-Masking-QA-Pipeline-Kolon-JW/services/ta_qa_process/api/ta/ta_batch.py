import json
import pandas as pd
import time
import pymysql
import datetime

from typing import Union
from concurrent import futures
from openai import OpenAI
from collections import defaultdict
from dotenv import load_dotenv
from api.ta.ta_model import VoCModel
import re

MODEL_NAME = "gpt-4o-mini"
MAX_RETRIES = 5


def contains_no_korean(text):
    return not re.search(r"[\uac00-\ud7a3]", text)


def check_result(result, total_usage, task_func):
    """
    TA 결과가 유효한지 확인하고 재실행에 대한 GPT usage를 수합하는 함수.
    한글로 생성되어야 하는 TA 항목의 결과에 한글이 없으면 할루시네이션이라고 판단하고 Exception을 raise 해서 재실행되도록 함.
    """

    content = str(result["result"]["content"])
    if contains_no_korean(content) and content.lower() != "none":
        raise Exception(
            f"GPT result is not Korean: '{content}', need to re-execute {task_func.__name__}."
        )

    cur_usage = result["result"]["usage"]

    total_usage = {k: v + cur_usage[k] for k, v in total_usage.items()}
    result["result"]["usage"] = total_usage

    return result, total_usage


def execute_with_retries(task_func, data, logger, sleep=5):
    """
    Retry max 3 times in case of OpenAI API error or parse error
    """
    # return task_func(data)

    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    total_usage2 = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }  # sentiment

    retries = 0
    while retries < MAX_RETRIES:
        try:
            result = task_func(data)
            # usage = result["result"]["usage"]
            # total_usage = {k: v + usage[k] for k, v in total_usage.items()}
            # content = str(result["result"]["content"])
            # if contains_no_korean(content) and content.lower() != "none":
            #     raise Exception(
            #         f"GPT result is not Korean: '{content}', need to re-execute {str(task_func)}."
            #     )
            # result["result"]["usage"] = total_usage

            if task_func.__name__ == "category_sentiment":  # type(result) == tuple
                category, sentiment = result

                category_updated, total_usage = check_result(category, total_usage, task_func)
                sentiment_updated, total_usage2 = check_result(sentiment, total_usage2, task_func)

                result = (category_updated, sentiment_updated)
            else:
                result_updated, total_usage = check_result(result, total_usage, task_func)
                result = result_updated

            break
        except Exception as exc:
            retries += 1
            if retries == MAX_RETRIES:
                logger.error(
                    f"All {MAX_RETRIES} retries failed for function {str(task_func.__name__)}, UID {data['UID']} Exception: {exc}. Returning 'ERROR' content."
                )
                # result = {
                #     "content": "ERROR",
                #     "raw": exc,
                #     "usage": total_usage,
                # }
                result = {
                    "UID": data["UID"],
                    "task": task_func.__name__,
                    "result": {
                        "content": "GPT ERROR",
                        "raw": str(exc),
                        "usage": total_usage,
                    },
                }
            time.sleep(sleep)

    return result


def main(dataset: dict, logger=None):
    """
    Make all TA contents for daily batch input (board data)
    """
    global category_data, category_time
    now = datetime.datetime.now()
    if (now - category_time).total_seconds() > 86400:
        logger.info(f"Time since last update > 24 hours: updating category data. Last update at: {category_time}, current time: {now}")
        category_data, category_time = _call_db_procedure()
        logger.info(f"Category classification data updated:\n{category_data}")
    else:
        logger.info(f"Time since last update < 24 hours: no update needed. Last update at: {category_time}, current time: {now}")

    client = OpenAI()
    model = VoCModel(openai_client=client, model_name=MODEL_NAME, category=category_data, logger=logger)

    tasks = [
        model.category_sentiment,  # category_classification과 sentiment_classification 합침
        # model.category_classification,
        model.speaker_summarization,
        model.brand_extraction,
        model.title_generation,
        # model.content_summarization,
        # model.sentiment_classification,
        model.keyword_extraction,
    ]

    results = []

    with futures.ThreadPoolExecutor(max_workers=len(tasks)) as executor:
        futures_to_func = {}
        for key, datapoint in dataset.items():
            for func in tasks:
                futures_to_func[
                    executor.submit(execute_with_retries, func, datapoint, logger)
                ] = func

        for future in futures.as_completed(futures_to_func):
            output = future.result()
            func = futures_to_func[future]
            # results.append(output)
            func_name = func.__name__
            if type(output) == tuple:
                # for category_sentiment, there are two results in tuple
                results.extend(list(output))
            else:
                results.append(output)

    results = align_results(results, logger)

    return results


def align_results(result_list, logger):
    # Initialize a defaultdict of dictionaries
    result = defaultdict(dict)

    # Validate input type
    if not isinstance(result_list, list):
        logger.error(
            "Invalid input: Expected a list, got %s", type(result_list).__name__
        )
        return {}

    for i, res in enumerate(result_list):
        # Validate each element is a dictionary
        if not isinstance(res, dict):
            logger.error(
                "Invalid element at index %d: Expected a dictionary, got %s , value %s",
                i,
                type(res).__name__,
                res,
            )
            continue

        # Ensure required keys are present
        if "UID" not in res:
            logger.error("Missing 'UID' key in element at index %d", i)
            continue
        if "task" not in res:
            logger.error("Missing 'task' key in element at index %d", i)
            continue
        if "result" not in res:
            logger.error("Missing 'result' key in element at index %d", i)
            continue

        key = res["UID"]

        # Handle unexpected types for 'task' and 'result' values
        task = res["task"]
        result_value = res["result"]

        if not isinstance(task, str):
            logger.error(
                "Invalid 'task' value at index %d: Expected a string, got %s",
                i,
                type(task).__name__,
            )
            continue

        # Add result to the dictionary
        result[key][task] = result_value

    return result



def _call_db_procedure():
    # Maria_DB_HOST = '172.19.112.18' # prod
    Maria_DB_HOST = '172.19.112.132' # dev
    Maria_DB_PORT = 3306
    Maria_DB_USER = 'saas3002'
    Maria_DB_PW = '@saas3002'
    conn = pymysql.connect(
        host=Maria_DB_HOST,
        port=Maria_DB_PORT,
        user=Maria_DB_USER,
        password=Maria_DB_PW,
        database="saas3002"
    )
    cur = conn.cursor()
    sql = "CALL spn_couns_type_s_v01(@out_code, @out_msg)"
    cur.execute(sql)
    res = cur.fetchall()
    now = datetime.datetime.now()
    return res, now

category_data, category_time = _call_db_procedure()