# Module Imports
import mariadb
from datetime import datetime, timedelta
import sys
import json
import os
from pytz import timezone
import json
import re
import traceback

KST = timezone("Asia/Seoul")

Maria_DB_HOST = "172.19.112.132"
# Maria_DB_HOST = '172.19.112.18'
Maria_DB_PORT = 3306
Maria_DB_USER = "saas3002"
Maria_DB_PW = "@saas3002"

# QA_TITLE_TO_COLUMN_MAPPING = {
#     "모니터링>고객 컴플레인": "Q01",
#     "서비스마인드>경청": "Q02",
#     "서비스마인드>공감": "Q03",
#     "서비스마인드>예절": "Q04",
#     "소프트스킬>고객정보확인": "Q05",
#     "소프트스킬>본인확인": "Q06",
#     "소프트스킬>언어습관": "Q07",
#     "소프트스킬>추가문의확인": "Q08",
#     "오프닝>첫인사": "Q09",
#     "질문·문제해결>문의내용파악": "Q10",
#     "질문·문제해결>필수사항안내": "Q11",
#     "클로징>끝인사": "Q12"
# }

QA_TITLE_TO_COLUMN_MAPPING = {
    "오프닝>첫인사": "Q01",
    "소프트스킬>언어습관": "Q02",
    "소프트스킬>고객정보확인": "Q03",
    "소프트스킬>본인확인": "Q04",
    "소프트스킬>추가문의확인": "Q05",
    "질문·문제해결>문의내용파악": "Q06",
    "질문·문제해결>필수사항안내": "Q07",
    "서비스마인드>경청": "Q08",
    "서비스마인드>공감": "Q09",
    "서비스마인드>예절": "Q10",
    "클로징>끝인사": "Q11",
    "모니터링>고객 컴플레인": "Q12",
}


def parse_stt_data_sentence(raw_record):
    if raw_record["SUCCESS_YN"]:
        current_call_sentence_list = []
        call_epoch_time = float(raw_record["UID"].split("-")[-1])
        call_datetime = datetime.fromtimestamp(call_epoch_time)

        for idx, sent in enumerate(raw_record["STT_OUTPUT_LIST"]):
            curr_sentence = {}
            curr_sentence["UID"] = raw_record["UID"]
            curr_sentence["STT_SEQ"] = idx
            curr_sentence["SPEAKER"] = sent["channel"]
            curr_sentence["DATA"] = sent["masked_text"]
            curr_sentence["CALL_START_DATE"] = sent["start_time"]
            curr_sentence["CALL_END_DATE"] = sent["end_time"]

            curr_sentence["UPDATED_AT"] = (
                datetime.now().astimezone(KST).strftime("%Y-%m-%d %H:%M:%S")
            )
            curr_sentence["CREATED_AT"] = raw_record["CDATE"]

            current_call_sentence_list.append(curr_sentence)
        return current_call_sentence_list
    else:
        return None


def insert_record_to_sentences_table(record, cursor, logger):
    current_call_sentence_list = parse_stt_data_sentence(record)

    # Check if the parsed list is not empty or None
    if current_call_sentence_list is not None and len(current_call_sentence_list) > 0:
        first_record_dict = current_call_sentence_list[0]
        placeholders = ", ".join(["%s"] * len(first_record_dict))
        columns = ", ".join(first_record_dict.keys())
        sql = "REPLACE INTO tb_stt_lm ( %s ) VALUES ( %s )" % (columns, placeholders)

        # Insert each record individually
        for sentence in current_call_sentence_list:
            values = list(sentence.values())
            try:
                cursor.execute(sql, values)
            except mariadb.Error as e:
                # Log the error and the problematic record
                logger.error(
                    "Exception occurred when inserting into tb_stt_lm: %s", str(e)
                )
                logger.error("UID: %s", str(record["UID"]))
                logger.error("Failed record: %s", str(sentence))
                # Raise the exception to be handled by the parent caller
                raise e


def parse_stt_ta_data(raw_record):
    """Parse data from the current record into a standardized event dictionary."""

    # Initialize parsed_data with default values
    parsed_data = {
        "PROJ_CD": raw_record["PROJ_CD"],
        "CDATE": raw_record["CDATE"],
        "CHANNEL_TYPE": raw_record["CHANNEL_TYPE"],
        "UID": raw_record["UID"],
        "CONTENT": None,  # Default to None, will update if SUCCESS_YN is True
        "CREATED_AT": datetime.now().astimezone(KST).strftime("%Y-%m-%d %H:%M:%S"),
        "CREATED_BY": "BATCH",
        "ID": "",  # Default to empty string, will update if CHANNEL_TYPE is "CHAT"
    }

    # Update content if SUCCESS_YN is True
    if raw_record.get("SUCCESS_YN", False):
        parsed_data["CONTENT"] = raw_record.get("MASKED_CONTENT")

    # Update ID if CHANNEL_TYPE is "CHAT"
    if parsed_data["CHANNEL_TYPE"] == "CHAT":
        parsed_data["ID"] = raw_record.get("ID", "")

    return parsed_data


def insert_recored_to_stt_ta_table(record, cursor, logger):
    parsed_data = parse_stt_ta_data(record)

    placeholders = ", ".join(["%s"] * len(parsed_data))
    columns = ", ".join(parsed_data.keys())
    sql = "REPLACE INTO tb_ai_ta_data ( %s ) VALUES ( %s )" % (columns, placeholders)

    try:
        cursor.execute(sql, list(parsed_data.values()))

    # Just pass the exception to parent caller
    except mariadb.Error as e:
        logger.error("Exception occur when insert into tb_ai_ta_data: %s", str(e))
        logger.error("UID: %s", str(record["UID"]))
        raise


def insert_multiple_stt_records(records, logger):
    logger.info(
        f"Got {len(records)} STT/TA call results. Start parsing and insert to DB"
    )

    try:
        conn = mariadb.connect(
            user=Maria_DB_USER,
            password=Maria_DB_PW,
            host=Maria_DB_HOST,
            port=Maria_DB_PORT,
            database="saas3002",
        )
        cursor = conn.cursor()

        count = 0

        for record in records:
            try:
                count += 1
                insert_record_to_sentences_table(record, cursor, logger)
                conn.commit()
                # insert_recored_to_stt_ta_table(record, cursor, logger)
                # conn.commit()
            except Exception as e:
                logger.error("Exception: %s", str(e))
                logger.error("UID: %s", str(record["UID"]))
                logger.error("record: %s", str(record))

        print(f"Insert {count} records to DB")
        logger.info(f"Finished inserting {count} records to sentences_table tb_stt_lm DB")

    except mariadb.Error as e:
        logger.error("Exception occurred when connecting to MariaDB: %s", str(e))
        raise

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def fetch_batch_ta_target(in_gb, in_cdate, logger=None):
    """
    Call the stored procedure spn_ta_target_s_v01 to fetch records from the database based on the provided UID.

    Parameters:
        in_gb (str): The type of call ('CALL' or 'CHAT').
        in_cdate (str): The datetime to filter records by. eg. 20240824
        logger (logging.Logger, optional): Logger for error logging.

    Returns:
        tuple: A tuple containing:
            - list: A list of records matching the UID.
            - str: The output code from the stored procedure.
            - str: The output message from the stored procedure.
    """
    try:
        conn = mariadb.connect(
            user=Maria_DB_USER,
            password=Maria_DB_PW,
            host=Maria_DB_HOST,
            port=Maria_DB_PORT,
            database="saas3002",
        )
        cursor = conn.cursor()

        # Define the call to the stored procedure
        sql = "CALL spn_ta_batch_target_s_v01(%s, %s, @out_code, @out_msg)"
        params = (in_gb, in_cdate)

        # Execute the stored procedure
        cursor.execute(sql, params)

        # Fetch the results if any
        result_list = []
        while True:
            if cursor.description:
                result = cursor.fetchall()
                if result:
                    # Get column names from the cursor
                    column_names = [desc[0] for desc in cursor.description]
                    # Convert results to a list of dictionaries
                    # result_list.extend([dict(zip(column_names, record)) for record in result])

                    # Cast CDATE from datetime type to string
                    result_list.extend(
                        [
                            {
                                key: (str(value) if key == "CDATE" else value)
                                for key, value in zip(column_names, record)
                            }
                            for record in result
                        ]
                    )
            if not cursor.nextset():
                break

        # Retrieve the output parameters
        cursor.execute("SELECT @out_code, @out_msg")
        out_code_value, out_msg_value = cursor.fetchone()

        return result_list, out_code_value, out_msg_value

    except mariadb.Error as e:
        if logger:
            logger.error(
                "Failed to fetch records from the database: %s\nTraceback: %s",
                str(e),
                traceback.format_exc(),
                extra={"UID": in_cdate},
            )
        return None, None, None

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def parse_ta_data(raw_record):
    """Parse TA data from the current record into a standardized event dictionary."""

    # Initialize parsed_data with default values
    parsed_data = {
        "PROJ_CD": raw_record["PROJ_CD"],
        "SITE_CD": raw_record["SITE_CD"],
        "CDATE": raw_record["CDATE"],
        "CHANNEL_TYPE": raw_record["CHANNEL_TYPE"],
        "IO_DIVI": raw_record["IO_DIVI"],
        "UID": raw_record["UID"],
        "CALL_SEQ": raw_record["CALL_SEQ"],
        "TA_FLAG": raw_record["TA_FLAG"],
        "BRAND": None,
        "CONTENT": None,
        "TITLE": None,
        "CONTENT_SUMMARY": None,
        "CONTENT_CUSTOMER": None,
        "CONTENT_AGENT": None,
        # 'COUNS_NM_PATH': None,
        "COUNS_NM_L": None,
        "COUNS_NM_L_PROB": None,
        "COUNS_NM_M": None,
        "COUNS_NM_M_PROB": None,
        "COUNS_NM_S": None,
        "COUNS_NM_S_PROB": None,
        "POSITIVE_DEGREE": None,
        "KEYWORD_1": None,
        "KEYWORD_2": None,
        "KEYWORD_3": None,
        "KEYWORD_4": None,
        "KEYWORD_5": None,
        "TOPIC": None,
        "TOPIC_SUM_1": None,
        "TOPIC_SUM_2": None,
        "SENTIMENT_CLS": None,
        "QA_FLAG": None,
        "AW_FLAG": "Y",
        "BATCH_FLAG": None,
        "ETC1": None,
        "ETC2": None,
        "ETC3": None,
        "ETC4": None,
        "ETC5": None,
        "UPDATED_BY": "SYSTEM",
    }

    # Update fields if SUCCESS_YN is True
    if raw_record.get("SUCCESS_YN", False):
        parsed_data.update(
            {
                "CONTENT": raw_record.get("CONTENT"),
                "CONTENT_CUSTOMER": raw_record.get("CONTENT_CUSTOMER"),
                "CONTENT_AGENT": raw_record.get("CONTENT_AGENT"),
                # 'COUNS_NM_PATH': raw_record.get('COUNS_NM_PATH')
                "COUNS_NM_L": raw_record.get("COUNS_NM_L"),
                "COUNS_NM_L_PROB": raw_record.get("COUNS_NM_L_PROB"),
                "COUNS_NM_M": raw_record.get("COUNS_NM_M"),
                "COUNS_NM_M_PROB": raw_record.get("COUNS_NM_M_PROB"),
                "COUNS_NM_S": raw_record.get("COUNS_NM_S"),
                "COUNS_NM_S_PROB": raw_record.get("COUNS_NM_S_PROB"),
                "KEYWORD_1": raw_record.get("KEYWORD_1"),
                "KEYWORD_2": raw_record.get("KEYWORD_2"),
                "KEYWORD_3": raw_record.get("KEYWORD_3"),
                "SENTIMENT_CLS": raw_record.get("SENTIMENT_CLS"),
                "BRAND": (
                    raw_record.get("BRAND")
                    if raw_record.get("BRAND") != "None"
                    else None
                ),
                "TITLE": (
                    raw_record.get("TITLE")
                    if raw_record.get("TITLE") != "None"
                    else None
                ),
            }
        )

        brand_mappings = {
            "이사칠": "24/7",
            "WAAC": "왁"
        }

        # Update parsed_data['BRAND'] if it exists in the brand_mappings
        parsed_data['BRAND'] = brand_mappings.get(parsed_data['BRAND'], parsed_data['BRAND'])

    return parsed_data


def insert_batch_ta_record(ai_ta_records, logger):

    inserted_success_count = 0
    try:
        conn = mariadb.connect(
            user=Maria_DB_USER,
            password=Maria_DB_PW,
            host=Maria_DB_HOST,
            port=Maria_DB_PORT,
            database="saas3002",
        )
        cursor = conn.cursor()

        sql = """
            CALL spn_ta_u_v01(
                %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, @1, @2
            )
        """

        for record in ai_ta_records:

            try:

                parsed_data = parse_ta_data(record)

                params = (
                    parsed_data["PROJ_CD"],
                    parsed_data["SITE_CD"],
                    parsed_data["CDATE"],
                    parsed_data["CHANNEL_TYPE"],
                    parsed_data["IO_DIVI"],
                    parsed_data["UID"],
                    parsed_data["CALL_SEQ"],
                    parsed_data["BRAND"],
                    parsed_data["CONTENT"],
                    parsed_data["TITLE"],
                    parsed_data["CONTENT_SUMMARY"],
                    parsed_data["CONTENT_CUSTOMER"],
                    parsed_data["CONTENT_AGENT"],
                    # parsed_data['COUNS_NM_PATH'],
                    parsed_data["COUNS_NM_L"],
                    parsed_data["COUNS_NM_L_PROB"],
                    parsed_data["COUNS_NM_M"],
                    parsed_data["COUNS_NM_M_PROB"],
                    parsed_data["COUNS_NM_S"],
                    parsed_data["COUNS_NM_S_PROB"],
                    parsed_data["POSITIVE_DEGREE"],
                    parsed_data["KEYWORD_1"],
                    parsed_data["KEYWORD_2"],
                    parsed_data["KEYWORD_3"],
                    parsed_data["KEYWORD_4"],
                    parsed_data["KEYWORD_5"],
                    parsed_data["TOPIC"],
                    parsed_data["TOPIC_SUM_1"],
                    parsed_data["TOPIC_SUM_2"],
                    parsed_data["SENTIMENT_CLS"],
                    parsed_data["TA_FLAG"],
                    parsed_data["QA_FLAG"],
                    parsed_data["AW_FLAG"],
                    parsed_data["BATCH_FLAG"],
                    parsed_data["ETC1"],
                    parsed_data["ETC2"],
                    parsed_data["ETC3"],
                    parsed_data["ETC4"],
                    parsed_data["ETC5"],
                    parsed_data["UPDATED_BY"],
                )

                cursor.execute(sql, params)

                # Retrieve the output parameters
                cursor.execute("SELECT @out_code, @out_msg")
                out_code_value, out_msg_value = cursor.fetchone()

                formatted_sql = sql % params
                logger.info(f"Executing SQL command: {formatted_sql}")

                inserted_success_count += 1
            except mariadb.Error as e:
                logger.error(
                    f"Error executing procedure for record: {record}, Error: {str(e)}"
                )
                conn.rollback()  # Rollback the transaction if an error occurs in executing the procedure
                continue  # Proceed with the next record

        conn.commit()
    except mariadb.Error as e:
        logger.error(f"Database connection or operation error: {str(e)}")
        conn.rollback()  # Rollback if a connection or other operation fails

    except Exception as e:
        logger.error(
            "Exception: Failed to fetch records from the database: %s\nTraceback: %s",
            str(e),
            traceback.format_exc(),
        )
        conn.rollback()  # Rollback if a connection or other operation fails

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

        return inserted_success_count


def replace_none_string(value):
    if isinstance(value, list) and not value:
        return None
    return None if value in ("None", "") else value


def parse_qa_data(raw_record):
    parsed_data = {}
    parsed_data["PROJ_CD"] = raw_record["PROJ_CD"]
    parsed_data["SITE_CD"] = raw_record["SITE_CD"]
    parsed_data["CDATE"] = raw_record["CDATE"]
    parsed_data["CHANNEL_TYPE"] = raw_record["CHANNEL_TYPE"]

    parsed_data["CONTENT"] = raw_record["CONTENT"]

    parsed_data["IO_DIVI"] = raw_record["IO_DIVI"]
    parsed_data["UID"] = raw_record["UID"]
    parsed_data["CALL_SEQ"] = raw_record["CALL_SEQ"]

    if raw_record["SUCCESS_YN"]:

        # Add the filling code here for Q01 -> Q12
        for title, column_prefix in QA_TITLE_TO_COLUMN_MAPPING.items():
            if title in raw_record["QA_RESULTS"]:
                content = replace_none_string(
                    raw_record["QA_RESULTS"][title]["content"]
                )

                # Fill in the main assessment
                parsed_data[column_prefix] = replace_none_string(
                    content.get("assessment", None)
                )

                # Fill in the comment for REASON and SPEAK (assuming they are the same as per your description)
                parsed_data[f"{column_prefix}_REASON"] = replace_none_string(
                    content.get("comment", None)
                )
                parsed_data[f"{column_prefix}_SPEAK"] = replace_none_string(
                    content.get("remark", None)
                )

        parsed_data["UPDATED_BY"] = "SYSTEM"
    else:
        parsed_data["AW_FLAG"] = "Y"
        parsed_data["UPDATED_BY"] = "SYSTEM"
    return parsed_data


def insert_batch_qa_record(ai_qa_records, logger):
    """
    Inserts a batch of QA records into the database by calling the stored procedure `spn_ai_qa_u_v01`.

    This function processes a list of AI QA records, prepares the data according to the expected
    input parameters of the stored procedure `spn_ai_qa_u_v01`, and then inserts each record
    into the database. The function handles potential database errors and logs them accordingly.

    """
    try:

        inserted_success_count = 0

        conn = mariadb.connect(
            user=Maria_DB_USER,
            password=Maria_DB_PW,
            host=Maria_DB_HOST,
            port=Maria_DB_PORT,
            database="saas3002",
        )
        cursor = conn.cursor()

        sql = """
            CALL spn_ai_qa_u_v01(
                %s, %s,
                %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s,
                @out_code, @out_msg
            )
        """

        for record in ai_qa_records:
            try:
                parsed_data = parse_qa_data(record)

                params = (
                    parsed_data["PROJ_CD"],
                    parsed_data["UID"],
                    parsed_data.get("Q01", None),
                    parsed_data.get("Q01_REASON", None),
                    parsed_data.get("Q01_SPEAK", None),
                    parsed_data.get("Q02", None),
                    parsed_data.get("Q02_REASON", None),
                    parsed_data.get("Q02_SPEAK", None),
                    parsed_data.get("Q03", None),
                    parsed_data.get("Q03_REASON", None),
                    parsed_data.get("Q03_SPEAK", None),
                    parsed_data.get("Q04", None),
                    parsed_data.get("Q04_REASON", None),
                    parsed_data.get("Q04_SPEAK", None),
                    parsed_data.get("Q05", None),
                    parsed_data.get("Q05_REASON", None),
                    parsed_data.get("Q05_SPEAK", None),
                    parsed_data.get("Q06", None),
                    parsed_data.get("Q06_REASON", None),
                    parsed_data.get("Q06_SPEAK", None),
                    parsed_data.get("Q07", None),
                    parsed_data.get("Q07_REASON", None),
                    parsed_data.get("Q07_SPEAK", None),
                    parsed_data.get("Q08", None),
                    parsed_data.get("Q08_REASON", None),
                    parsed_data.get("Q08_SPEAK", None),
                    parsed_data.get("Q09", None),
                    parsed_data.get("Q09_REASON", None),
                    parsed_data.get("Q09_SPEAK", None),
                    parsed_data.get("Q10", None),
                    parsed_data.get("Q10_REASON", None),
                    parsed_data.get("Q10_SPEAK", None),
                    parsed_data.get("Q11", None),
                    parsed_data.get("Q11_REASON", None),
                    parsed_data.get("Q11_SPEAK", None),
                    parsed_data.get("Q12", None),
                    parsed_data.get("Q12_REASON", None),
                    parsed_data.get("Q12_SPEAK", None),
                    parsed_data.get("Q13", None),
                    parsed_data.get("Q13_REASON", None),
                    parsed_data.get("Q13_SPEAK", None),
                    parsed_data.get("Q14", None),
                    parsed_data.get("Q14_REASON", None),
                    parsed_data.get("Q14_SPEAK", None),
                    parsed_data.get("Q15", None),
                    parsed_data.get("Q15_REASON", None),
                    parsed_data.get("Q15_SPEAK", None),
                    parsed_data.get("Q16", None),
                    parsed_data.get("Q16_REASON", None),
                    parsed_data.get("Q16_SPEAK", None),
                    parsed_data.get("Q17", None),
                    parsed_data.get("Q17_REASON", None),
                    parsed_data.get("Q17_SPEAK", None),
                    parsed_data.get("Q18", None),
                    parsed_data.get("Q18_REASON", None),
                    parsed_data.get("Q18_SPEAK", None),
                    parsed_data.get("Q19", None),
                    parsed_data.get("Q19_REASON", None),
                    parsed_data.get("Q19_SPEAK", None),
                    parsed_data.get("Q20", None),
                    parsed_data.get("Q20_REASON", None),
                    parsed_data.get("Q20_SPEAK", None),
                )

                cursor.execute(sql, params)

                # Retrieve the output parameters
                cursor.execute("SELECT @out_code, @out_msg")
                out_code_value, out_msg_value = cursor.fetchone()

                logger.info(
                    f"Executed procedure with out_code: {out_code_value}, out_msg: {out_msg_value}"
                )

                # formatted_sql = sql % params
                # logger.info(f"Executing SQL command: {formatted_sql}")

                inserted_success_count += 1

            except mariadb.Error as e:
                logger.error(
                    f"Error executing procedure for record: {record}, Error: {str(e)}"
                )
                formatted_sql = sql % params
                logger.error(f"Error executing procedure for record: {formatted_sql}")

                conn.rollback()  # Rollback the transaction if an error occurs in executing the procedure
                continue  # Proceed with the next record

        conn.commit()
    except mariadb.Error as e:
        logger.error(f"Database connection or operation error: {str(e)}")
        conn.rollback()  # Rollback if a connection or other operation fails

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

        return inserted_success_count


def insert_gpt_cost_record(records, logger):

    try:
        conn = mariadb.connect(
            user=Maria_DB_USER,
            password=Maria_DB_PW,
            host=Maria_DB_HOST,
            port=Maria_DB_PORT,
            database="saas3002",
        )
        cursor = conn.cursor()

        for record in records:

            placeholders = ", ".join(["%s"] * len(record))

            columns = ", ".join(record.keys())

            sql = "INSERT INTO %s ( %s ) VALUES ( %s )" % (
                "tb_gpt_cost",
                columns,
                placeholders,
            )
            cursor.execute(sql, list(record.values()))

        conn.commit()

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()



def fetch_post_remain_target(in_gb, in_cdate, logger=None):
    """
    Call the stored procedure spn_ta_post_remain_target_s_v01 to fetch UIDs from the database.

    Parameters:
        in_gb (str): The type of call ('CALL' or 'CHAT').
        in_cdate (str): The datetime to filter records by. eg. 20240910
        logger (logging.Logger, optional): Logger for error logging.

    Returns:
        tuple: A tuple containing:
            - list: A list of UIDs.
            - str: The output code from the stored procedure.
            - str: The output message from the stored procedure.
    """
    try:
        conn = mariadb.connect(
            user=Maria_DB_USER,
            password=Maria_DB_PW,
            host=Maria_DB_HOST,
            port=Maria_DB_PORT,
            database="saas3002",
        )
        cursor = conn.cursor()

        # Define the call to the stored procedure
        sql = "CALL spn_ta_post_remain_target_s_v01(%s, %s, @out_code, @out_msg)"
        params = (in_gb, in_cdate)

        # Execute the stored procedure
        cursor.execute(sql, params)

        # Fetch the results (list of UIDs)
        uid_list = []
        while True:
            if cursor.description:
                result = cursor.fetchall()
                if result:
                    # Only one column "UID", so just extract the values
                    uid_list.extend([record[0] for record in result])
            if not cursor.nextset():
                break

        # Retrieve the output parameters
        cursor.execute("SELECT @out_code, @out_msg")
        out_code_value, out_msg_value = cursor.fetchone()

        return uid_list, out_code_value, out_msg_value

    except mariadb.Error as e:
        if logger:
            logger.error(
                "Failed to fetch UIDs from the database: %s\nTraceback: %s",
                str(e),
                traceback.format_exc(),
                extra={"UID": in_cdate},
            )
        return None, None, None

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
