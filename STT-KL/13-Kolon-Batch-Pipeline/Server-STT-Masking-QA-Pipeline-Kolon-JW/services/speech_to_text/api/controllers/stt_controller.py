# This file will contain the code for handling incoming API requests and invoking the model for prediction.


from flask import request
from flask_restful import Resource
from api.models import STTModel  # Assuming the model class is defined in the models folder
from api.services import PredictionService  # Import the service responsible for model prediction
from api.utils.logger import setup_logger, save_result_to_json
import datetime
import os
import traceback
# Define a global variable to store the loaded model
loaded_model = None

class PredictionController(Resource):

    def __init__(self):
        self.logger = setup_logger()


    def transcribe(self):

        data = request.get_json()  # Assuming the request payload contains a dictionary in JSON format
        data_input_folder = data.get('data_input')  # Extract the input data from the request payload
        data_output_folder = data.get('data_output_folder')  # Extract the input data from the request payload

        self.logger.info("Received request: %s", data_input_folder)  # Log the received request

        global loaded_model  # Access the global variable

        if loaded_model is None:
            # Load the trained model
            loaded_model = STTModel()
            # loaded_model.load_model()

        # Create output folder if not exist
        os.makedirs(data_output_folder, exist_ok=True)

        # Create an instance of the prediction service and invoke the prediction method
        try:
            prediction_service = PredictionService(loaded_model)
            result = prediction_service.transcribe(data_input_folder, data_output_folder, self.logger)
        except Exception as e:
            self.logger.error(f"Exception: STT Service failed to transcribe with exception:  {str(e)}\nTraceback: {traceback.format_exc()}\n Request data: {data}")
            return {'error': 'Internal Server Error: Failed to save results to AICC DB.'}, 500

        response = {'result': result}

        self.logger.info("Sending response output: %s", data_output_folder)  # Log the response

        return response, 200

    def transcribe_gpu(self, use_kenlm=False):

        data = request.get_json()  # Assuming the request payload contains a dictionary in JSON format
        data_input_folder = data.get('data_input')  # Extract the input data from the request payload
        data_output_folder = data.get('data_output_folder')  # Extract the input data from the request payload

        self.logger.info("Received request: %s", data_input_folder)  # Log the received request

        global loaded_model  # Access the global variable

        if loaded_model is None:
            # Load the trained model
            loaded_model = STTModel()
            # loaded_model.load_model()

        # Create output folder if not exist
        os.makedirs(data_output_folder, exist_ok=True)

        # Create an instance of the prediction service and invoke the prediction method
        try:
            prediction_service = PredictionService(loaded_model)
            if use_kenlm:
                result = prediction_service.transcribe_gpu_lm(data_input_folder, data_output_folder, self.logger)
            else:
                result = prediction_service.transcribe_gpu(data_input_folder, data_output_folder, self.logger)
        except Exception as e:
            self.logger.error(f"Exception: STT Service failed to transcribe with exception:  {str(e)}\nTraceback: {traceback.format_exc()}\n Request data: {data}")
            return {'error': 'Internal Server Error: Failed to save results to AICC DB.'}, 500

        response = {'result': result}

        self.logger.info("Sending response output: %s", data_output_folder)  # Log the response

        return response, 200


    