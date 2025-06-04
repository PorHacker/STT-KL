from flask import Flask
from api.routes import register_routes
from dotenv import load_dotenv
from gevent.pywsgi import WSGIServer
import os

# Load environment variables from .env file
load_dotenv()

import config

import logging
from logging.handlers import RotatingFileHandler

# Define the log directory and file path
log_directory = "logs"
os.makedirs(log_directory, exist_ok=True)
log_file1 = os.path.join(log_directory, "gevent_access.log")
log_file2 = os.path.join(log_directory, "gevent_error.log")

# Set up loggers for access and error logs
access_logger = logging.getLogger('gevent_access')
access_logger.setLevel(logging.DEBUG)
access_log_handler = logging.FileHandler(log_file1)
access_log_formatter = logging.Formatter('%(asctime)s - %(message)s')
access_log_handler.setFormatter(access_log_formatter)
access_logger.addHandler(access_log_handler)

error_logger = logging.getLogger('gevent_error')
error_logger.setLevel(logging.DEBUG)
error_log_handler = logging.FileHandler(log_file2)
error_log_formatter = logging.Formatter('%(asctime)s - %(message)s [in %(pathname)s:%(lineno)d]')
error_log_handler.setFormatter(error_log_formatter)
error_logger.addHandler(error_log_handler)

def create_app():
    # Create the Flask application object
    app = Flask(__name__)

    # Load the configuration based on environment
    app.config.from_object(config.CURRENT_CONFIG)

    # Register routes and blueprints
    register_routes(app)

    return app


# Create the Flask application
app = create_app()

# Run the development server if executed directly
if __name__ == '__main__':
    # Serve the app using gevent.pywsgi.WSGIServer
    wsgi_server = WSGIServer(('0.0.0.0', 5004), app, log=access_logger, error_log=error_logger)
    wsgi_server.serve_forever()