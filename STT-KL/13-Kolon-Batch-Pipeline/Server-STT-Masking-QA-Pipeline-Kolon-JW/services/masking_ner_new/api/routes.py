from flask import Blueprint, jsonify, request
from api.call_masking import main as call_masking_main
from api.chat_masking import main as chat_masking_main
from api.utils.logger import setup_logger

# Create a Blueprint object for the service1 routes
masking_ner_bp = Blueprint('masking_ner', __name__, url_prefix='/masking_ner')

# Initialize logger
logger = setup_logger()

# Define the routes and their handlers
@masking_ner_bp.route('/masking_call', methods=['POST'])
def masking_all():
    request_data = request.get_json()


    total_conversation_dict = request_data["total_conversation_dict"]

    # Log the request data
    logger.info(f"Request data len: {len(total_conversation_dict)}")

    call_ids = list(total_conversation_dict.keys())
    conversation_list = [total_conversation_dict[x] for x in call_ids]

    masked_results = call_masking_main(conversation_list, logger)

    masked_results_dict = {}
    for idx in range(0, len(call_ids)):
        masked_results_dict[call_ids[idx]] = {
            "text" : conversation_list[idx],
            "masking_text" : masked_results[idx]
        }

    # Log the response data
    logger.info(f"Call Masking NER Response data: {masked_results_dict}")

    return jsonify(masked_results_dict)


# Define the routes and their handlers
@masking_ner_bp.route('/masking_chat', methods=['POST'])
def masking_chat():
    request_data = request.get_json()


    total_conversation_dict = request_data["total_conversation_dict"]

    # Log the request data
    logger.info(f"Request data len: {len(total_conversation_dict)}")

    call_ids = list(total_conversation_dict.keys())
    conversation_list = [total_conversation_dict[x] for x in call_ids]

    masked_results = chat_masking_main(conversation_list, logger)

    masked_results_dict = {}
    for idx in range(0, len(call_ids)):
        masked_results_dict[call_ids[idx]] = {
            "text" : conversation_list[idx],
            "masking_text" : masked_results[idx]
        }

    # Log the response data
    logger.info(f"CHAT Masking NER Response data: {masked_results_dict}")

    return jsonify(masked_results_dict)

# Health check endpoint
@masking_ner_bp.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200  # just means we're on air

# Register the service1 Blueprint in the Flask application
def register_routes(app):
    app.register_blueprint(masking_ner_bp)
