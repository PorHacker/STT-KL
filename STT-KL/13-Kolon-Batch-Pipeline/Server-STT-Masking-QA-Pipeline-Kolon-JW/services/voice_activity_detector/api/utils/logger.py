import logging
import datetime
import os
import json

def setup_logger():
    logger = logging.getLogger("my_logger")
    logger.setLevel(logging.INFO)

    # Define the log directory and file path
    log_directory = "logs"
    os.makedirs(log_directory, exist_ok=True)
    log_file = os.path.join(log_directory, "request_response.log")

    # Create a basic file handler (logrotate will handle rotation)
    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(handler)

    return logger

def setup_logger_by_day():
    logger = logging.getLogger("my_logger")
    logger.setLevel(logging.INFO)

    # Create a timestamp for the log file name and folder structure
    now = datetime.datetime.now()
    year = now.strftime("%Y")
    month = now.strftime("%m")
    day = now.strftime("%d")

    # Create the folder structure if it doesn't exist
    log_directory = f"logs/{year}-{month}/{day}"
    os.makedirs(log_directory, exist_ok=True)

    # Create a timestamp for the log file name
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S.%f")

    # Define the log file path with the timestamp
    log_file = f"{log_directory}/request_response.log"

    # Create a file handler and set the logging format
    file_handler = logging.FileHandler(log_file)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    return logger, now


def save_result_to_json(result):
    # Create a timestamp for the log file name and folder structure
    now = datetime.datetime.now()
    year = now.strftime("%Y")
    month = now.strftime("%m")
    day = now.strftime("%d")

    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S.%f")


    # Create the folder structure if it doesn't exist
    output_directory = f"outputs/{year}-{month}/{day}"
    os.makedirs(output_directory, exist_ok=True)

    # Define the result file path with the timestamp
    result_file = f"{output_directory}/{timestamp}.json"

    # Save the result as JSON
    with open(result_file, "w") as f:
        json.dump(result, f)

    return result_file
