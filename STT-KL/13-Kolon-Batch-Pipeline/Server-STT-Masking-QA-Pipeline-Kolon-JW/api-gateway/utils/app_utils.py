import os
import stat
import uuid
from datetime import datetime, timedelta
import json
import traceback
from utils.database import insert_gpt_cost_record, insert_batch_ta_record, insert_batch_qa_record
import time
from pytz import timezone
import requests
import re
KST = timezone('Asia/Seoul')


def is_valid_request_payload(payload):
    required_fields = ['request_id', "in_gb"]
    return all(field in payload for field in required_fields)


def generate_request_id(request_datetime):
    timestamp = request_datetime.strftime("%Y-%m-%d_%H-%M-%S.%f")
    request_id = f'{timestamp}_{uuid.uuid4()}'

    return request_id


def generate_upload_location(upload_dir, request_datetime, upload_request_id):
    upload_request_folder = os.path.join(upload_dir, upload_request_id)
    if not os.path.exists(upload_request_folder):
        os.makedirs(upload_request_folder)
        os.chmod(upload_request_folder, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
    return upload_request_folder


def generate_ta_qa_location(ta_output_dir, upload_request_id):
    ta_qa_request_folder = os.path.join(ta_output_dir, upload_request_id)
    if not os.path.exists(ta_qa_request_folder):
        os.makedirs(ta_qa_request_folder)
        os.chmod(ta_qa_request_folder, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
    return ta_qa_request_folder



def save_ta_results(text_conversation_records, ta_request_payload, ta_process_response, in_gb, logger):
    """Save TA results to the database."""
    try:
        insert_ai_ta_records = []
        gpt_usage_records = []

        for record in text_conversation_records:
            CALL_ID = record.get("UID")
            UNIQUE_BOARD_CALL_ID = record["UNIQUE_UID"]

            if UNIQUE_BOARD_CALL_ID in ta_request_payload:
                # Update MASKED CONTENT in record
                record["MASKED_CONTENT"] = ta_request_payload[UNIQUE_BOARD_CALL_ID]["MASKED_CONTENT"]

                cur_insert_ai_ta_record = prepare_ta_insert_record(record, ta_process_response[UNIQUE_BOARD_CALL_ID], in_gb)
                insert_ai_ta_records.append(cur_insert_ai_ta_record)

                gpt_usage_records.extend(prepare_gpt_cost_records(ta_process_response[UNIQUE_BOARD_CALL_ID], record, in_gb))

        success_count = insert_batch_ta_record(insert_ai_ta_records, logger)
        insert_gpt_cost_record(gpt_usage_records, logger)
        insert_message = f"Inserted TA records: Success {success_count}/{len(insert_ai_ta_records)}"
        logger.info(insert_message)

        return insert_ai_ta_records, gpt_usage_records, insert_message

    except Exception as e:
        logger.error("Failed to save TA results to DB: %s\nTraceback: %s", str(e), traceback.format_exc())
        raise



def save_qa_results(text_conversation_records, qa_request_payload, qa_process_response, in_gb, logger):
    """Save QA results to the database."""
    try:
        insert_ai_qa_records = []
        gpt_usage_records = []

        for record in text_conversation_records:
            CALL_ID = record.get("UID")
            UNIQUE_BOARD_CALL_ID = record["UNIQUE_UID"]

            if UNIQUE_BOARD_CALL_ID in qa_request_payload:
                # Update MASKED CONTENT in record
                record["MASKED_CONTENT"] = qa_request_payload[UNIQUE_BOARD_CALL_ID]["MASKED_CONTENT"]

                cur_insert_ai_qa_record = prepare_qa_insert_record(record, qa_process_response[UNIQUE_BOARD_CALL_ID], in_gb)
                insert_ai_qa_records.append(cur_insert_ai_qa_record)

                gpt_usage_records.extend(prepare_gpt_cost_records(qa_process_response[UNIQUE_BOARD_CALL_ID], record, in_gb))

        success_count = insert_batch_qa_record(insert_ai_qa_records, logger)
        insert_gpt_cost_record(gpt_usage_records, logger)
        insert_message = f"Inserted QA records: Success {success_count}/{len(insert_ai_qa_records)}"
        logger.info(insert_message)

        return insert_ai_qa_records, gpt_usage_records, insert_message

    except Exception as e:
        error_message = "Failed to save QA results to DB: %s\nTraceback: %s" % (str(e), traceback.format_exc())
        logger.error(error_message)
        raise Exception(error_message)


def prepare_ta_insert_record(record, ta_response, in_gb):
    """Prepare record for TA insertion."""

    default_content = None
    default_content_dict = {"customer": None, "agent": None}
    
    # Keywords
    ta_response_contents = ta_response
    keyword_extraction = ta_response_contents.get('keyword_extraction', {})
    keyword_content_list = keyword_extraction.get('content', [])

    cur_insert_ai_ta_record = {
        "SUCCESS_YN": True,
        "PROJ_CD": record['PROJ_CD'],
        "SITE_CD": record['SITE_CD'],
        "CDATE": record['CDATE'],
        "CHANNEL_TYPE": record['CHANNEL_TYPE'],
        "IO_DIVI": record['IO_DIVI'],
        "UID": record['UID'],
        "CALL_SEQ": record['CALL_SEQ'],
        "TA_FLAG": record['TA_FLAG'], 
        "CONTENT": record['MASKED_CONTENT'],

        # "CONTENT_SUMMARY": ta_response_contents.get('content_summarization', {}).get('content', default_content),
        "CONTENT_CUSTOMER": ta_response_contents.get('speaker_summarization', {}).get('content', default_content_dict).get('customer', default_content),
        "CONTENT_AGENT": ta_response_contents.get('speaker_summarization', {}).get('content', default_content_dict).get('agent', default_content),
        
        "COUNS_NM_L": ta_response_contents.get('category_classification', {}).get('content', {}).get('depth1', default_content),
        "COUNS_NM_L_PROB": ta_response_contents.get('category_classification', {}).get('content', {}).get('depth1_prob', None),
        "COUNS_NM_M": ta_response_contents.get('category_classification', {}).get('content', {}).get('depth2', default_content),
        "COUNS_NM_M_PROB": ta_response_contents.get('category_classification', {}).get('content', {}).get('depth2_prob', default_content),
        "COUNS_NM_S": ta_response_contents.get('category_classification', {}).get('content', {}).get('depth3', default_content),
        "COUNS_NM_S_PROB": ta_response_contents.get('category_classification', {}).get('content', {}).get('depth3_prob', default_content),
        
        "KEYWORD_1": keyword_content_list[0] if len(keyword_content_list) > 0 else None,
        "KEYWORD_2": keyword_content_list[1] if len(keyword_content_list) > 1 else None,
        "KEYWORD_3": keyword_content_list[2] if len(keyword_content_list) > 2 else None,

        "BRAND": ta_response_contents.get('brand_extraction', {}).get('content', default_content),
        "TITLE": ta_response_contents.get('title_generation', {}).get('content', default_content),
        "SENTIMENT_CLS": ta_response_contents.get('sentiment_classification', {}).get('content', default_content),
    }


    # Don't need to update CONTENT and TA_FLAG for BOARD since it sync directly into target table  => 2024/09/10 : Updated, insert CONTENT 
    
    if in_gb in ["BOARD"]:
        cur_insert_ai_ta_record.update({
            # "CONTENT": None,  # Updated 2024/09/10
            "TA_FLAG": None, 
        })

    
    return cur_insert_ai_ta_record


def prepare_qa_insert_record(record, qa_response, in_gb):
    """Prepare record for QA insertion."""
    cur_insert_ai_qa_record = {
        "SUCCESS_YN": True,
        "PROJ_CD": record['PROJ_CD'],
        "SITE_CD": record['SITE_CD'],
        "CDATE": record['CDATE'],
        "CHANNEL_TYPE": record['CHANNEL_TYPE'],
        "IO_DIVI": record['IO_DIVI'],
        "UID": record['UID'],
        "CALL_SEQ": record['CALL_SEQ'],
        "QA_RESULTS": qa_response,
        "CONTENT": record['MASKED_CONTENT'],

    }
    return cur_insert_ai_qa_record


def prepare_gpt_cost_records(response, record, in_gb):
    """Prepare GPT cost records for insertion."""
    gpt_usage_records = []
    for task_name, task_value in response.items():
        cost_record = {
            "PROJ_CD": record.get('PROJ_CD'),
            "GB": in_gb,
            "GB2": task_name,
            "UID": record.get('UID'),
            "INPUT_TOKEN": task_value.get('usage', {}).get('prompt_tokens', 0),
            "OUTPUT_TOKEN": task_value.get('usage', {}).get('completion_tokens', 0),
            "CREATED_AT": datetime.now().astimezone(KST).strftime("%Y-%m-%d %H:%M:%S"),
            "CREATED_BY": "TA-QA-BATCH"
        }
        gpt_usage_records.append(cost_record)
    return gpt_usage_records


def check_duplicate_uid(text_conversation_records):
    # Step 1: Create a dictionary to count occurrences of each UID
    preprocess_message = "  "
    uid_count = {}
    for record in text_conversation_records:
        uid = str(record["UID"]) + "-" + str(record["CALL_SEQ"])  ## 2024.10.08 Update Unique UID = UID + CALL_SEQ
        if uid in uid_count:
            uid_count[uid].append(record)
        else:
            uid_count[uid] = [record]

    # Step 2: Print out records with duplicate UIDs
    for uid, records in uid_count.items():
        if len(records) > 1:
            preprocess_message += f"| Duplicate UID: {uid}"

    return preprocess_message

    
# Function to send a POST request and return the elapsed time
def send_realtime_post_process_request(in_gb, call_chat_uid, logger):

    
    request_datetime = datetime.now()
    request_id = generate_request_id(request_datetime)

    POST_PROCESS_SERVER_URL = "http://10.13.6.240:8031"
    if in_gb == "CALL":
        api_url = f"{POST_PROCESS_SERVER_URL}/process_call"
        payload = {
            "request_id": request_id + "_REMAIN_CALL",
            "call_uid": str(call_chat_uid)
        }
    elif in_gb == "CHAT":
        api_url = f"{POST_PROCESS_SERVER_URL}/process_chat"
        payload = {
            "request_id": request_id + "_REMAIN_CHAT",
            "chat_room_uid": str(call_chat_uid)
        }

    start_time = time.time()  # Record start time

    try:
        response = requests.post(api_url, json=payload)
        elapsed_time = time.time() - start_time  # Calculate elapsed time

        logger.info(f"Request ID: {request_id},in_gb: {in_gb} UID: {call_chat_uid}, Response Status: {response.status_code}, Response Time: {elapsed_time:.2f}s")

        if response.status_code != 200:
            logger.error(f"Failed to process UID: {call_chat_uid}. Status code: {response.status_code}, Response: {response.text}")
            raise Exception(f"Failed to process UID: {call_chat_uid}. Status code: {response.status_code}")

        return elapsed_time, response.json()

    except requests.RequestException as e:
        logger.error(f"Error occurred while sending POST request for UID: {call_chat_uid}. Error: {str(e)}")
        raise


def process_correct_branch_name(raw_data):
    """
    주어진 상담문을 처리하여 잘못된 단어를 교체하고 중복된 단어를 제거합니다.
    :param raw_data: str, 처리할 상담문
    :return: str, 처리된 상담문
    """
    try:
    # 교체할 단어와 정규 표현식 정의
        replacements = {
            r'\b(?:고롱|코로|코론|코롱|코른|코름|코오|코오롱|코오프|코응|코트리|코트프를)(을|를|에서|에게|입니다|로|에)?\b': lambda m: '코오롱' + (m.group(1) if m.group(1) else ''),
            r'\b(?:더같은|더카|더카은|더카톡|더카트)(을|를|에서|에게|입니다|로|에)?\b': lambda m: '더카트' + (m.group(1) if m.group(1) else ''),
            r'\b(?:코롱몰|코오록물|코오롱몬|코오롱몰|코오롱몰에|코오롱물|코오르몬)(을|를|에서|에게|입니다|로|에)?\b': lambda m: '코오롱몰' + (m.group(1) if m.group(1) else ''),
            r'\b(?:헬리코튼|헬리콥튼)(을|를|에서|에게|입니다|로|에)?\b': lambda m: '헨리코튼' + (m.group(1) if m.group(1) else ''),
            r'\b(?:브랜드우드)(을|를|에서|에게|입니다|로|에)?\b': lambda m: '브렌우드' + (m.group(1) if m.group(1) else ''),
            r'\b(?:러키슈웨트|러키슈트|러키슈이트|러키슈에트)(을|를|에서|에게|입니다|로|에)?\b': lambda m: '럭키슈에뜨' + (m.group(1) if m.group(1) else ''),
            r'\b(?:슈커마보니|스커마보니|수커마보니|슈구마보니|시구마보니|슈퍼마보니)(을|를|에서|에게|입니다|로|에)?\b': lambda m: '슈콤마보니' + (m.group(1) if m.group(1) else ''),
            r'\b(?:생니클라우스|데이클라우스|백리클라우스)(을|를|에서|에게|입니다|로|에)?\b': lambda m: '잭니클라우스' + (m.group(1) if m.group(1) else ''),
            r'\b(?:커스튼멜로우)(을|를|에서|에게|입니다|로|에)?\b': lambda m: '커스텀멜로우' + (m.group(1) if m.group(1) else ''),
            r'\b(?:캔브리지)(을|를|에서|에게|입니다|로|에)?\b': lambda m: '캠브리지' + (m.group(1) if m.group(1) else ''),
            r'\b(?:캔브리지멤버스)(을|를|에서|에게|입니다|로|에)?\b': lambda m: '캠브리지멤버스' + (m.group(1) if m.group(1) else '')
        }

        # 전체 상담문에서 교체 적용
        processed_text = raw_data
        for pattern, replacement in replacements.items():
            processed_text = re.sub(pattern, replacement, processed_text)

        # 중복된 단어를 한 번만 남기기
        processed_text = re.sub(r'(코오롱)\s*(\1)+', '코오롱', processed_text)
        processed_text = re.sub(r'(더카트)\s*(\1)+', '더카트', processed_text)
        processed_text = re.sub(r'(코오롱몰)\s*(\1)+', '코오롱몰', processed_text)
        processed_text = re.sub(r'(헨리코튼)\s*(\1)+', '헨리코튼', processed_text)
        processed_text = re.sub(r'(브렌우드)\s*(\1)+', '브렌우드', processed_text)
        processed_text = re.sub(r'(럭키슈에뜨)\s*(\1)+', '럭키슈에뜨', processed_text)
        processed_text = re.sub(r'(슈콤마보니)\s*(\1)+', '슈콤마보니', processed_text)
        processed_text = re.sub(r'(잭니클라우스)\s*(\1)+', '잭니클라우스', processed_text)
        processed_text = re.sub(r'(커스텀멜로우)\s*(\1)+', '커스텀멜로우', processed_text)
        processed_text = re.sub(r'(캠브리지)\s*(\1)+', '캠브리지', processed_text)
        processed_text = re.sub(r'(캠브리지멤버스)\s*(\1)+', '캠브리지멤버스', processed_text)

        return processed_text
    
    except Exception as e:
        # 로그 작성 또는 에러 처리
        return raw_data
