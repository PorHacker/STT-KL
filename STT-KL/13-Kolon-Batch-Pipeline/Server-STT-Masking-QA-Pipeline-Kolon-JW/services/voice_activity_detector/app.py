from flask import Flask
from api.routes import register_routes
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

import config

def create_app():
    # Create the Flask application object
    app = Flask(__name__)
    # Register routes and blueprints
    register_routes(app)

    app.config.from_object(config.CURRENT_CONFIG)


    return app


# Create the Flask application
app = create_app()

# Run the development server if executed directly
if __name__ == '__main__':
    # app.run()
    app.run(host='0.0.0.0')
