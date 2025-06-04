from flask import Blueprint, jsonify, request
from .controllers import PredictionController

# Create a Blueprint object for the service1 routes
vad_bp = Blueprint('vad', __name__, url_prefix='/vad')

# Instantiate the UserController
vad_controller = PredictionController()

# Define the routes and their handlers
@vad_bp.route('/split', methods=['POST'])
def predict_vad():
    # Call the appropriate function in the PredictionController to get topic
    topic_results = vad_controller.split_speaker()
    return jsonify(topic_results)

@vad_bp.route('/split_paralell', methods=['POST'])
def predict_vad_paralell():
    # Call the appropriate function in the PredictionController to get topic
    topic_results = vad_controller.split_speaker_paralell()
    return jsonify(topic_results)


@vad_bp.route('/split_paralell_pa', methods=['POST'])
def predict_vad_paralell_pa():
    # Call the appropriate function in the PredictionController to get topic
    topic_results = vad_controller.pyanno_split_speaker_paralell()
    return jsonify(topic_results)

# Health check endpoint
@vad_bp.route('/health')
def health_check():
    return jsonify({'status': 'healthy'})  # just means we're on air

# Register the service1 Blueprint in the Flask application
def register_routes(app):
    app.register_blueprint(vad_bp)
