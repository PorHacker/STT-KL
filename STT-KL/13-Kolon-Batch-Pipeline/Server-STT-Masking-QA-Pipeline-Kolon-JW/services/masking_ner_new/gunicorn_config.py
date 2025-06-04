# gunicorn_config.py

bind = "0.0.0.0:5003"  # Bind the server to this address and port
timeout  = 86400
workers = 16  # Number of worker processes
# Add logging settings
loglevel = "info"
errorlog = "logs/gunicorn_error.log"
accesslog = "logs/gunicorn_access.log"