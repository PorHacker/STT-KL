"""
총 10가지 항목 평가 진행. 
각 데이터와 평가항목이 독립적이므로 데이터셋 크기 * 항목 개수 만큼의 task를 만들어서 multithread로 실행하는 코드. 
"""

import os
import json
import re
import pandas as pd
import time
import string

from openai import OpenAI
from dotenv import load_dotenv
from concurrent import futures
from typing import Union
from collections import defaultdict
from datetime import datetime

from api.qa.qa_model import QAModel
from collections import defaultdict

# QA uses gpt-4o model
MODEL_NAME = "gpt-4o"
PROMPT_COST = 5
COMPLETION_COST = 15

# MODEL_NAME = "gpt-4o-mini"
# PROMPT_COST = 0.15
# COMPLETION_COST = 0.6

MAX_RETRIES = 3
RETRY_DELAY = 5

# TODO: validate token numbers to avoid gpt model input token limit error
# TODO: reexecute if the generated remark does not exist in input text


def contains_no_korean(text):
    return not re.search(r"[\uac00-\ud7a3]", text)


def execute_with_retries(func, input_data, logger):
    """Retry max 3 times in case of OpenAI API error or syntax error while parsing"""

    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    retries = 0
    while retries < MAX_RETRIES:
        # result = func(input_data)
        # break
        try:
            result = func(input_data)
            usage = result["result"]["usage"]
            total_usage = {k: v + usage[k] for k, v in total_usage.items()}
            content = str(result["result"]["content"])
            if contains_no_korean(content):
                logger.warning(
                    f"GPT results contains_no_korean in {retries} retries. Content: {content}"
                )
                raise Exception(
                    f"GPT result is not Korean: [{content}], need to re-execute."
                )
            result["result"]["usage"] = total_usage
            break
        except Exception as exc:
            logger.error(
                f"All {MAX_RETRIES} retries failed for function... Exception: {exc}. Returning 'ERROR'."
            )
            retries += 1
            if retries == MAX_RETRIES:
                result = {
                    "UID": input_data["UID"],
                    "task": func,
                    "result": {
                        # "content": "ERROR",
                        "content": "GPT ERROR",
                        "raw": str(exc),
                        "usage": total_usage,
                    },
                }
            else:
                time.sleep(RETRY_DELAY)

    return result


def main(dataset: dict, logger):
    client = OpenAI()
    model = QAModel(client=client, model_name=MODEL_NAME, logger=logger)

    results = []

    with futures.ThreadPoolExecutor() as executor:
        futures_to_func = {}
        for datapoint in dataset.values():
            qa_tasks = model.assessment_func_list()
            for task in qa_tasks:
                futures_to_func[
                    executor.submit(execute_with_retries, task, datapoint, logger)
                ] = task

        for future in futures.as_completed(futures_to_func):
            result = future.result()
            results.append(result)

    results = align_results(results, logger)

    return results


def align_results(result_list, logging):
    # Initialize a defaultdict of dictionaries
    result = defaultdict(dict)

    # Validate input type
    if not isinstance(result_list, list):
        logging.error(
            "Invalid input: Expected a list, got %s", type(result_list).__name__
        )
        return {}

    for i, res in enumerate(result_list):
        # Validate each element is a dictionary
        if not isinstance(res, dict):
            logging.error(
                "Invalid element at index %d: Expected a dictionary, got %s",
                i,
                type(res).__name__,
            )
            continue

        # Ensure required keys are present
        if "UID" not in res:
            logging.error("Missing 'UID' key in element at index %d", i)
            continue
        if "task" not in res:
            logging.error("Missing 'task' key in element at index %d", i)
            continue
        if "result" not in res:
            logging.error("Missing 'result' key in element at index %d", i)
            continue

        key = res["UID"]

        # Handle unexpected types for 'task' and 'result' values
        task = res["task"]
        result_value = res["result"]

        if not isinstance(task, str):
            logging.error(
                "Invalid 'task' value at index %d: Expected a string, got %s",
                i,
                type(task).__name__,
            )
            continue

        # Add result to the dictionary
        result[key][task] = result_value

    return result