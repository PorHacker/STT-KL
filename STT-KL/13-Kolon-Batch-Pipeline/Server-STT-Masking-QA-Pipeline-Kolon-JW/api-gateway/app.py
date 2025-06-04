from flask import Flask, request, jsonify, make_response
import requests
import threading
from datetime import datetime, timedelta
import json
from dotenv import load_dotenv
import time

# Load environment variables from .env file
load_dotenv()
import os
import time
import config
import shutil
from utils.logger import setup_logger
from utils.database import insert_multiple_stt_records, fetch_batch_ta_target, fetch_post_remain_target
import traceback
from utils.text_preprocess import preprocess
from utils.app_utils import *

logger = setup_logger()

lock = threading.Lock()

microservices = {
    # VAD Services
    "voice_activity_detector": {
        "url": "http://aithe-sti-vad:5001",
        "endpoint": "/vad/split_paralell_pa",
    },
    # STT Services
    "speech_to_text": {
        "url": "http://aithe-sti-asr:5004",
        "endpoint": "/asr/transcribe",
    },
    "speech_to_text_gpu": {
        "url": "http://aithe-sti-asr:5004",
        "endpoint": "/asr/transcribe-gpu",
    },
    "speech_to_text_gpu_lm": {
        "url": "http://aithe-sti-asr:5004",
        "endpoint": "/asr/transcribe-gpu-lm",
    },
    # Masking & NER Services
    "call_masking_ner": {
        "url": "http://aithe-sti-masking-ner:5003",
        "endpoint": "/masking_ner/masking_call",
    },
    "chat_board_masking_ner": {
        "url": "http://aithe-sti-masking-ner:5003",
        "endpoint": "/masking_ner/masking_chat",
    },
    # TA QA Services
    "ta_process": {
        "url": "http://aithe-sti-ta_qa_process:5004",
        "endpoint": "/generative-ai/ta_process",
    },
    "qa_process": {
        "url": "http://aithe-sti-ta_qa_process:5004",
        "endpoint": "/generative-ai/qa_process",
    },
}


def create_app():
    # Create the Flask application object
    app = Flask(__name__)
    app.config.from_object(config.CURRENT_CONFIG)
    return app


app = create_app()


@app.errorhandler(Exception)
def handle_global_error(e):

    # Log the error with contextual information
    logger.error(
        "Unhandled Exception: %s\nTraceback: %s",
        str(e),
        traceback.format_exc(),
        extra={
            "url": request.url,
            "user_agent": request.headers.get("User-Agent"),
        },
    )

    # Return an error response to the client
    return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500


@app.route("/request-upload", methods=["POST"])
def request_upload():

    request_datetime = datetime.now()
    # Generate a unique request ID
    request_id = generate_request_id(request_datetime)

    # Generate the upload location
    upload_location = generate_upload_location(
        app.config["UPLOAD_DIR"], request_datetime, request_id
    )

    # Store the request ID and upload location for future reference

    # Prepare the response JSON with the request ID and upload location
    response = {"request_id": request_id, "upload_location": upload_location}

    # Return the response JSON
    return jsonify(response)


@app.route("/process_call", methods=["POST"])
def process_call():
    request_data = request.json

    # Acquire the lock, blocking other threads until the lock is released
    lock.acquire()

    try:
        # Retrieve the request ID and any other relevant information from the completion notification
        request_id = request_data.get("request_id")
        uploaded_location = request_data.get("uploaded_location")

        if not request_id or not uploaded_location:
            logger.error(
                "Bad Request: Missing required parameters 'request_id' or 'uploaded_location'"
            )
            return (
                jsonify(
                    {
                        "error": "Bad Request: Missing 'request_id' or 'uploaded_location'"
                    }
                ),
                400,
            )

        data_input_folder = uploaded_location

        start_time = time.time()
        logger.info(f"process_call: {data_input_folder} at {start_time}")

        # Step 1: Voice Activity Detection Service
        vad_output_folder = data_input_folder.replace(
            "input_request_folder", "output_vad_folder"
        )
        vad_input = {
            "data_input_folder": data_input_folder,
            "data_output_folder": vad_output_folder,
        }
        vad_response = forward_request("voice_activity_detector", vad_input)
        if not vad_response or "result" not in vad_response[0]:
            logger.error(
                "Internal Server Error: VAD Service failed to process the request.",
                extra={
                    "url": request.url,
                    "user_agent": request.headers.get("User-Agent"),
                    "request_data": request.json,
                },
            )
            return (
                jsonify(
                    {
                        "error": "Internal Server Error: VAD Service failed to process the request."
                    }
                ),
                500,
            )

        elif len(vad_response[0]["result"]) == 0:

            logger.error(
                "Internal Server Error: VAD Service failed to process the request. Empty input request folder!",
                extra={
                    "url": request.url,
                    "user_agent": request.headers.get("User-Agent"),
                    "request_data": request.json,
                },
            )
            return (
                jsonify(
                    {
                        "error": "Internal Server Error: VAD Service failed to process the request. Empty input request folder!"
                    }
                ),
                422,
            )

        final_output_json_path = os.path.join(
            vad_output_folder, f"{request_id}_vad_check_he.json"
        )
        with open(final_output_json_path, "w", encoding="utf-8") as fp:
            json.dump(vad_response[0]["result"], fp, ensure_ascii=False, indent=4)

        vad_end_time = time.time()
        logger.info(f"VAD took:{vad_end_time - start_time}")

        # Step 2: Speech to Text Service
        stt_output_folder = data_input_folder.replace(
            "input_request_folder", "output_stt_ta_folder"
        )
        stt_input = {
            "data_input": vad_response[0]["result"],
            "data_output_folder": stt_output_folder,
        }
        stt_response = forward_request("speech_to_text_gpu", stt_input)
        if not stt_response:
            logger.error(
                "Internal Server Error: ASR Service failed to process the request.",
                extra={
                    "url": request.url,
                    "user_agent": request.headers.get("User-Agent"),
                    "request_data": request.json,
                },
            )
            return (
                jsonify(
                    {
                        "error": "Internal Server Error: ASR Service failed to process the request."
                    }
                ),
                500,
            )

        # Gather conversation
        total_conversation_dict = {}

        for record in stt_response[0]["result"]:
            call_id = record["call_id"]

            # Check if diraization or stt_recode success or not
            split_success = record["diarization_record"]["successYN"]
            stt_success = record["stt_record"]["successYN"]
            # Skip current call if diarization or stt false
            if not split_success or not stt_success:
                # logger.warning(f"Skip {record['call_id']} Reason: Split successYN {split_success} STT successYN {stt_success}")
                continue

            stt_output_list = record["stt_record"]["stt_engine_output"]
            total_conversation = ""

            for chunk_stt in stt_output_list:
                if "counselor" in chunk_stt["audio_filepath"].split("/")[-1]:
                    total_conversation += f"상담사: {chunk_stt['pred_text']}\n"
                else:
                    total_conversation += f"고객: {chunk_stt['pred_text']}\n"


            # Correct spelling branch name in first sentence
            total_conversation = process_correct_branch_name(total_conversation)
            total_conversation_dict[call_id] = total_conversation

        stt_end_time = time.time()
        logger.info(f"STT took:{stt_end_time - vad_end_time}")

        # Step 3: Masking NER process
        if len(total_conversation_dict) > 0:

            masking_ner_input = {
                "total_conversation_dict": total_conversation_dict,
                "request_id": request_id,
            }

            masking_ner_response = forward_request(
                "call_masking_ner", masking_ner_input
            )
            if not masking_ner_response:
                logger.error(
                    "Internal Server Error: Masking NER Service failed to process the request.",
                    extra={
                        "url": request.url,
                        "user_agent": request.headers.get("User-Agent"),
                        "request_data": request.json,
                    },
                )
                return (
                    jsonify(
                        {
                            "error": "Internal Server Error: Masking NER Service failed to process the request."
                        }
                    ),
                    500,
                )

            masking_ner_end_time = time.time()
            logger.info(f"Masking NER took:{masking_ner_end_time - stt_end_time}")
        else:
            logger.info(f"No conversation to process masking NER")

        try:
            logger.info("Collect results to insert DB")
            db_records = []

            for record in stt_response[0]["result"]:
                current_records = {}
                call_id = record["call_id"]

                current_records["PROJ_CD"] = call_id.split("_")[0]

                # Get call datetime
                call_epoch_time = float(call_id.split("-")[-1])
                call_datetime = datetime.fromtimestamp(call_epoch_time)
                current_records["CDATE"] = call_datetime.strftime("%Y-%m-%d %H:%M:%S")
                current_records["CHANNEL_TYPE"] = "CALL"
                current_records["UID"] = call_id.split("_")[-1]
                # Check if diraization or stt_recode success or not
                split_success = record["diarization_record"]["successYN"]
                stt_success = record["stt_record"]["successYN"]
                if not split_success or not stt_success:
                    current_records["SUCCESS_YN"] = False
                    current_records["CONTENT"] = (
                        f"SPLIT: {split_success}  | STT: {stt_success}"
                    )
                else:
                    current_records["SUCCESS_YN"] = True

                    current_records["CONTENT"] = total_conversation_dict[call_id]

                    masked_total_conversation = masking_ner_response[call_id]["masking_text"]

                    stt_output_list = record["stt_record"]["stt_engine_output"]

                    # Convert masked conversation to chunked stt output
                    chunks = masked_total_conversation.split("\n")
                    masked_chunk_stt_list = []

                    customer_masked_text = ""
                    counselor_masked_text = ""

                    for idx, chunk in enumerate(chunks):
                        chunk_text = chunk.strip()

                        if ": " in chunk_text:
                            speaker, text = chunk_text.split(": ", 1)
                            masked_chunk_stt_list.append(
                                {"speaker": speaker, "masked_text": text}
                            )

                            if speaker == "고객":
                                customer_masked_text += f"{text}\n"
                            elif speaker == "상담사":
                                counselor_masked_text += f"{text}\n"

                        else:
                            masked_chunk_stt_list.append(
                                {"speaker": "Unknown", "masked_text": chunk_text}
                            )
                            text = ""

                        if idx < len(stt_output_list):
                            stt_output_list[idx]["masked_text"] = text

                    current_records["STT_OUTPUT_LIST"] = stt_output_list
                    current_records["MASKED_CONTENT"] = masked_total_conversation
                # Save final masked conversation

                output_stt_ta_folder = data_input_folder.replace(
                    "input_request_folder", "output_stt_ta_folder"
                )
                final_output_json_path = os.path.join(
                    output_stt_ta_folder, f"{call_id}_final_output_masking.json"
                )
                with open(final_output_json_path, "w", encoding="utf-8") as fp:
                    json.dump(current_records, fp, ensure_ascii=False, indent=4)

                db_records.append(current_records)

            logger.info("Start insert results to db")
            insert_multiple_stt_records(db_records, logger)

        except Exception as e:
            logger.error(
                "Exception: Failed to save results to AICC DB %s\nTraceback: %s",
                str(e),
                traceback.format_exc(),
                extra={
                    "url": request.url,
                    "user_agent": request.headers.get("User-Agent"),
                    "request_data": request.json,
                },
            )
            return (
                jsonify(
                    {
                        "error": "Internal Server Error: Failed to save results to AICC DB."
                    }
                ),
                500,
            )

        # Delete all requested data
        # shutil.rmtree(data_input_folder)
        # shutil.rmtree(vad_output_folder)
        # shutil.rmtree(stt_output_folder)

        final_response = {
            "successYN": True,
            "request_id": request_id,
            "channel": "call",
            "msg": f"Successfully process calls and saved to DB",
        }
        return jsonify(final_response)

    finally:
        # Release the lock to allow other threads to acquire it
        lock.release()


@app.route("/process_ta_qa", methods=["POST"])
def process_ta_qa():
    try:
        request_data = request.json

        if not request_data or not is_valid_request_payload(request_data):
            logger.info(f"Return 400, Invalid request data request_data: {request_data}")
            return jsonify({"error": "Invalid request data"}), 400

        request_id = request_data["request_id"]
        in_gb = request_data["in_gb"]
        # request_cdate = request_data.get("cdate", datetime.now().strftime("%Y%m%d"))
        request_cdate = request_data.get(
            "cdate", datetime.now().strftime("%Y%m%d%H%M%S")
        )

        logger.info(f"Received Request with in_gb: {in_gb}, request_cdate: {request_cdate}")


        in_gb_possible_values = ["CALL", "CHAT", "BOARD"]
        if in_gb not in in_gb_possible_values:
            logger.error(
                f"Bad Request: Invalid value for in_gb. It must be one of {in_gb_possible_values}"
            )
            return (
                jsonify(
                    {
                        "error": f"Bad Request: Invalid value for in_gb. It must be one of {in_gb_possible_values}"
                    }
                ),
                400,
            )

        output_ta_qa_folder = generate_ta_qa_location(
            app.config["TA_DIR_DIR"], request_id
        )

        # Step 1: Get STT Conversation from DB
        text_conversation_records, fetch_error_codes, fetch_message = (
            fetch_batch_ta_target(in_gb, request_cdate, logger)
        )

        if not text_conversation_records:
            logger.info(f"No {in_gb} conversation records found in the database. request_data: {request_data}")
            return jsonify(
                {
                    "successYN": True,
                    "msg": f"No {in_gb} records found in the database.",
                    "text_conversation_records": text_conversation_records,
                }
            )

        fetch_end_time = time.time()
        logger.info(f"After Step 1: Fetch Data from DB with in_gb: {in_gb}, requested_uid: {request_cdate} DB text_conversation_records: {text_conversation_records}")


        # Preprocess and prepare payloads
        (
            ta_request_payload,
            qa_request_payload,
            masking_ner_request_payload,
            invalid_count,
        ) = preprocess_and_prepare_payloads(text_conversation_records, in_gb)

        preprocess_message = f"UID: {request_id} Request TA: {len(ta_request_payload)}/{len(text_conversation_records)} | QA : {len(qa_request_payload)}/{len(text_conversation_records)} | Invalid CONTENT: {invalid_count}"
        check_duplicate_uid_msg = check_duplicate_uid(text_conversation_records)
        preprocess_message += check_duplicate_uid_msg
        logger.info(preprocess_message)

        logger.info(f"After Step 2: Preprocess the text with in_gb: {in_gb}, requested_uid: {request_cdate}  ta_request_payload: {ta_request_payload}")
        logger.info(f"After Step 2: Preprocess the text with in_gb: {in_gb}, requested_uid: {request_cdate}  qa_request_payload: {qa_request_payload}")
        logger.info(f"After Step 2: Preprocess the text with in_gb: {in_gb}, requested_uid: {request_cdate}  masking_ner_request_payload: {masking_ner_request_payload}")


        # if len(ta_request_payload) == 0:
        #     final_response = {
        #         "successYN": True,
        #         "request_id": request_id,
        #         "in_gb": in_gb,
        #         "text_conversation_records": text_conversation_records,
        #         'msg': f"No valid {in_gb} conversation records found after preprocess!",
        #         "preprocess_msg" : preprocess_message,
        #     }
        #     return jsonify(final_response)

        # Step 3: Masking NER process (only for CHAT and BOARD, CALL already masked in post-processing)
        if in_gb in ["CHAT", "BOARD", "CALL"]:
            ta_request_payload, qa_request_payload, masking_ner_response = (
                apply_masking_ner(
                    masking_ner_request_payload,
                    ta_request_payload,
                    qa_request_payload,
                    request_id,
                    logger,
                    in_gb,
                )
            )
            final_output_json_path = os.path.join(
                output_ta_qa_folder, f"00_apply_masking_ner_{in_gb}.json"
            )
            with open(final_output_json_path, "w", encoding="utf-8") as fp:
                json.dump(masking_ner_response, fp, ensure_ascii=False, indent=4)


        logger.info(f"After Step 3: Masking NER process with in_gb: {in_gb}, requested_uid: {request_cdate}  ta_request_payload: {ta_request_payload}")
        logger.info(f"After Step 3: Masking NER process with in_gb: {in_gb}, requested_uid: {request_cdate}  qa_request_payload: {qa_request_payload}")
        logger.info(f"After Step 3: Masking NER process with in_gb: {in_gb}, requested_uid: {request_cdate}  masking_ner_response: {masking_ner_response}")

        # Step 4: TA Process
        insert_ai_ta_records, ta_gpt_usage_records, insert_ta_message = (
            [],
            [],
            "No TA processing",
        )  # Initialize variables to avoid UnboundLocalError

        # Only BOARD process TA
        if ta_request_payload and len(ta_request_payload) > 0:

            final_output_json_path = os.path.join(
                output_ta_qa_folder, f"01_ta_request_payload_{in_gb}.json"
            )
            with open(final_output_json_path, "w", encoding="utf-8") as fp:
                json.dump(ta_request_payload, fp, ensure_ascii=False, indent=4)

            ta_process_response = process_ta(ta_request_payload, request_id)

            logger.info(f"After Step 4: BOARD process TA with in_gb: {in_gb}, requested_uid: {request_cdate}  ta_process_response: {ta_process_response}")

            insert_ai_ta_records, ta_gpt_usage_records, insert_ta_message = (
                save_ta_results(
                    text_conversation_records,
                    ta_request_payload,
                    ta_process_response,
                    in_gb,
                    logger,
                )
            )

            logger.info(f"After Step 5: SAVE TA Results with in_gb: {in_gb}, requested_uid: {request_cdate}  insert_ai_ta_records: {insert_ai_ta_records}")
            logger.info(f"After Step 5: SAVE TA Results with in_gb: {in_gb}, requested_uid: {request_cdate}  ta_gpt_usage_records: {ta_gpt_usage_records}")



        # Step 5: QA Process
        insert_ai_qa_records, qa_gpt_usage_records, insert_qa_message = (
            [],
            [],
            "No QA processing",
        )  # Initialize variables to avoid UnboundLocalError
        if qa_request_payload and len(qa_request_payload) > 0:

            final_output_json_path = os.path.join(
                output_ta_qa_folder, f"02_qa_request_payload_{in_gb}.json"
            )
            with open(final_output_json_path, "w", encoding="utf-8") as fp:
                json.dump(qa_request_payload, fp, ensure_ascii=False, indent=4)
            
            qa_process_response = process_qa(qa_request_payload, request_id)

            logger.info(f"After Step 4: CALL process QA with in_gb: {in_gb}, requested_uid: {request_cdate}  qa_process_response: {qa_process_response}")

            insert_ai_qa_records, qa_gpt_usage_records, insert_qa_message = (
                save_qa_results(
                    text_conversation_records,
                    qa_request_payload,
                    qa_process_response,
                    in_gb,
                    logger,
                )
            )

            logger.info(f"After Step 5: SAVE QA Results with in_gb: {in_gb}, requested_uid: {request_cdate}  insert_ai_qa_records: {insert_ai_qa_records}")
            logger.info(f"After Step 5: SAVE QA Results with in_gb: {in_gb}, requested_uid: {request_cdate}  qa_gpt_usage_records: {qa_gpt_usage_records}")


        final_response = {
            "successYN": True,
            "request_id": request_id,
            "in_gb": in_gb,
            "text_conversation_records": text_conversation_records,
            "ta_task": {
                "insert_ai_ta_records": insert_ai_ta_records,
                "ta_gpt_usage_records": ta_gpt_usage_records,
                "insert_ta_message": insert_ta_message,
            },
            "qa_task": {
                "insert_ai_qa_records": insert_ai_qa_records,
                "qa_gpt_usage_records": qa_gpt_usage_records,
                "insert_qa_message": insert_qa_message,
            },
            "msg": f"Successfully processed {in_gb} and saved to DB",
            "preprocess_msg": preprocess_message,
        }

        final_output_json_path = os.path.join(
            output_ta_qa_folder, f"03_final_response_{in_gb}.json"
        )
        with open(final_output_json_path, "w", encoding="utf-8") as fp:
            json.dump(final_response, fp, ensure_ascii=False, indent=4)

        final_response = {
            "successYN": True,
            "request_id": request_id,
            "in_gb": in_gb,
            "ta_task" : {
                "insert_ta_message" : insert_ta_message,
            },
            "qa_task" : {
                "insert_qa_message": insert_qa_message,
            },
            'msg': f"Successfully processed {in_gb} and saved to DB",
            "preprocess_msg" : preprocess_message,
        }

        return jsonify(final_response)

    except KeyError as e:
        logger.error(f"Return 400, KeyError: Missing key in request data {str(e)}\nrequest_data:{request_data}")
        return (
            jsonify({"error": f"Bad Request: Missing key in request data, {str(e)}"}),
            400,
        )
    except Exception as e:
        logger.error(
            f"Return 500, Unexpected error: {str(e)}\nTraceback: {traceback.format_exc()}\nrequest_data:{request_data}"
        )
        return (
            jsonify(
                {
                    "error": f"Internal Server Error: {str(e)}",
                    "traceback": traceback.format_exc(),
                }
            ),
            500,
        )


def preprocess_and_prepare_payloads(text_conversation_records, in_gb):
    """Preprocess text and prepare payloads for TA, QA, and Masking NER processing."""
    ta_request_payload = {}
    qa_request_payload = {}
    masking_ner_request_payload = {}

    invalid_count = 0

    for record in text_conversation_records:
        UNIQUE_BOARD_CALL_ID = str(record.get("UID"))  + "-" +  str(record.get("CALL_SEQ"))
        record["UNIQUE_UID"] = UNIQUE_BOARD_CALL_ID
        call_content = record.get("CONTENT", "")

        if isinstance(record.get("CDATE"), datetime):
            record["CDATE"] = record["CDATE"].isoformat()

        try:

            if call_content:
                if record.get("TA_FLAG") == "Y" and in_gb == "BOARD":
                    content_stt_sentences = preprocess(text=call_content, in_gb=in_gb)

                    # Copy the record and remove "CONTENT" for ta_request_payload
                    payload_record = record.copy()

                    # Update UID to be UNIQUE
                    payload_record["UID"] = UNIQUE_BOARD_CALL_ID

                    # Pre assigned MASKED_CONTENT value, will be updated after NER
                    record["MASKED_CONTENT"] = content_stt_sentences
                    payload_record["MASKED_CONTENT"] = content_stt_sentences
                    del payload_record["CONTENT"]  # Remove CONTENT from the payload
                    ta_request_payload[UNIQUE_BOARD_CALL_ID] = payload_record

                    masking_ner_request_payload[UNIQUE_BOARD_CALL_ID] = content_stt_sentences

                if record.get("QA_FLAG") == "Y" and in_gb == "CALL":
                    content_stt_sentences = preprocess(text=call_content, in_gb=in_gb)
                    
                    # Copy the record and remove "CONTENT" for qa_request_payload
                    payload_record = record.copy()

                    # Update UID to be UNIQUE
                    payload_record["UID"] = UNIQUE_BOARD_CALL_ID

                    # Pre assigned MASKED_CONTENT value, will be updated after NER
                    record["MASKED_CONTENT"] = "\n".join(content_stt_sentences)
                    payload_record["MASKED_CONTENT"] = "\n".join(content_stt_sentences)
                    del payload_record["CONTENT"]  # Remove CONTENT from the payload
                    qa_request_payload[UNIQUE_BOARD_CALL_ID] = payload_record

                    masking_ner_request_payload[UNIQUE_BOARD_CALL_ID] = "\n".join(content_stt_sentences)

            else:
                invalid_count += 1
                logger.warning(f"{in_gb} ID: {UNIQUE_BOARD_CALL_ID} have None or empty content")

        except Exception as e:
            logger.error(
                "Error processing content for record %s: %s\nTraceback: %s",
                UNIQUE_BOARD_CALL_ID,
                str(e),
                traceback.format_exc(),
            )
            record["TA_FLAG"] = "N"
            record["QA_FLAG"] = "N"
            invalid_count += 1

    return (
        ta_request_payload,
        qa_request_payload,
        masking_ner_request_payload,
        invalid_count,
    )


def apply_masking_ner(
    masking_ner_request_payload,
    ta_request_payload,
    qa_request_payload,
    request_id,
    logger,
    in_gb,
):
    """Apply Masking NER process and update the TA request payload with masked content."""
    if masking_ner_request_payload:
        masking_ner_input = {
            "total_conversation_dict": masking_ner_request_payload,
            "request_id": request_id,
        }

        if in_gb in ["CHAT", "BOARD"]:
            masking_ner_response = forward_request(
                "chat_board_masking_ner", masking_ner_input
            )
            logger.info(f"Requested Masking NER for {in_gb} request_id: {request_id} response: {masking_ner_response}")
        else:
            logger.info(f"Send Masking NER for {in_gb} request_id: {request_id} masking_ner_input: {masking_ner_input}")
            masking_ner_response = forward_request(
                "call_masking_ner", masking_ner_input
            )
            logger.info(f"Requested Masking NER for {in_gb} request_id: {request_id} response: {masking_ner_response}")
        
        if not masking_ner_response:
            logger.error(
                "Internal Server Error: Masking NER Service failed to process the request."
            )
            logger.info(f"masking_ner_input: {masking_ner_input}")
            raise Exception("Masking NER Service failed to process the request.")

        # Update TA payload
        for key, record_data in ta_request_payload.items():
            if key in masking_ner_response:
                record_data["MASKED_CONTENT"] = masking_ner_response[key].get(
                    "masking_text", record_data["MASKED_CONTENT"]
                )

        # Update QA payload
        for key, record_data in qa_request_payload.items():
            if key in masking_ner_response:
                record_data["MASKED_CONTENT"] = masking_ner_response[key].get(
                    "masking_text", record_data["MASKED_CONTENT"]
                )

    else:
        logger.info("No conversation to process masking NER")
        raise Exception("No conversation to process masking NER.")

    return ta_request_payload, qa_request_payload, masking_ner_response


def process_ta(ta_request_payload, request_id):
    """Send TA payload for processing."""
    ta_process_input = {"data": ta_request_payload, "request_id": request_id}

    ta_process_response = forward_request("ta_process", ta_process_input)
    if not ta_process_response:
        logger.error(
            "Internal Server Error: TA Process Service failed to process the request."
        )
        logger.info(f"ta_request_payload: {ta_request_payload}")
        raise Exception("TA Process Service failed to process the request.")

    return ta_process_response


def process_qa(qa_request_payload, request_id):
    """Send QA payload for processing."""
    qa_process_input = {"data": qa_request_payload, "request_id": request_id}

    qa_process_response = forward_request("qa_process", qa_process_input)
    if not qa_process_response:
        error_message = "QA Process Service failed to process the request."
        logger.error(error_message)
        raise Exception(error_message)

    return qa_process_response


@app.route("/process_post_remain", methods=["POST"])
def process_post_remain():
    try:
        # Step 1: Parse incoming request data
        request_data = request.json

        if not request_data or not is_valid_request_payload(request_data):
            logger.info(f"POST Remain: Return 400, Invalid request data: {request_data}")
            return jsonify({"error": "Invalid request data"}), 400

        request_id = request_data["request_id"]
        in_gb = request_data["in_gb"]
        request_cdate = request_data.get("cdate", datetime.now().strftime("%Y%m%d"))

        output_ta_qa_folder = generate_ta_qa_location(
            app.config["TA_DIR_DIR"], request_id
        )


        in_gb_possible_values = ["CALL", "CHAT"]
        if in_gb not in in_gb_possible_values:
            logger.error(f"POST Remain: Bad Request: Invalid value for in_gb. Must be one of {in_gb_possible_values}")
            return jsonify({"error": f"Invalid value for in_gb. Must be one of {in_gb_possible_values}"}), 400

        logger.info(f"POST Remain: Received valid request data: in_gb={in_gb}, request_cdate={request_cdate}")

        # Step 2: Fetch the list of UIDs from the database
        uid_list, fetch_error_code, fetch_message = fetch_post_remain_target(in_gb, request_cdate, logger)

        if not uid_list:
            logger.info(f"POST Remain: No UIDs found for {in_gb} in the database. request_data: {request_data}")
            return jsonify({"successYN": True, "msg": f"No UIDs found for {in_gb} in the database."})

        logger.info(f"POST Remain: Fetched {len(uid_list)} UIDs from the database for {in_gb}.")
        logger.info(f"POST Remain: uid_list: {uid_list}")

        # Step 3: Send POST requests for each UID and log the elapsed time
        elapsed_times = []
        respond_datas = []
        for uid_record in uid_list:
            call_uid = uid_record
            if call_uid:
                try:
                    elapsed_time, respond_data = send_realtime_post_process_request(in_gb, call_uid, logger)
                    elapsed_times.append(elapsed_time)
                    respond_datas.append(respond_data)
                    logger.info(f"POST Remain: Successfully processed UID: {call_uid}, Elapsed time: {elapsed_time:.2f}s")
                except Exception as e:
                    logger.error(f"POST Remain: Error occurred while sending POST request for UID: {call_uid}, Error: {str(e)}")

        # Step 4: Final response
        if len(elapsed_times) > 0:
            logger.info(f"POST Remain: Completed processing {len(uid_list)} UIDs. Average elapsed time: {sum(elapsed_times) / len(elapsed_times):.2f}s")


        final_response = {
            "successYN": True,
            "msg": f"Successfully processed {in_gb} records.",
            "total_uids_processed": len(uid_list),
            "average_elapsed_time": sum(elapsed_times) / len(elapsed_times) if elapsed_times else 0,
            "respond_datas": respond_datas,
        }

        
        final_output_json_path = os.path.join(
            output_ta_qa_folder, f"02_final_response_{in_gb}.json"
        )
        with open(final_output_json_path, "w", encoding="utf-8") as fp:
            json.dump(final_response, fp, ensure_ascii=False, indent=4)


        del final_response["respond_datas"]

        return jsonify(final_response)

    except KeyError as e:
        logger.error(f"POST Remain: Return 400, KeyError: Missing key in request data {str(e)}. request_data: {request_data}")
        return jsonify({"error": f"Bad Request: Missing key in request data: {str(e)}"}), 400

    except Exception as e:
        logger.error(f"POST Remain: Return 500, Unexpected error: {str(e)}. Traceback: {traceback.format_exc()} request_data: {request_data}")
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500


# Health check endpoint
@app.route("/health")
def health_check():
    return jsonify({"status": "healthy"})


def forward_request(service_name, request_data):
    service = microservices.get(service_name)
    if not service:
        return None

    try:
        response = requests.post(
            service["url"] + service["endpoint"], json=request_data
        )
        if response.status_code == 200:
            return response.json()
        else:
            logger.warning(
                f"Warning: forward request to {service_name} return code: {response.status_code}"
            )
            return None
    except requests.exceptions.RequestException as e:
        logger.error(
            "Exception: Failed to forward request to %s with exception: %s\nTraceback: %s",
            service_name,
            str(e),
            traceback.format_exc(),
        )
        return None


if __name__ == "__main__":
    app.run(threaded=True, host="0.0.0.0")
