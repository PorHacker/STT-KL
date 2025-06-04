# gunicorn_config.py

bind = "0.0.0.0:5002"  # Bind the server to this address and port
# workers = 4  # Number of worker processes
timeout  = 86400
# workers = 2  # Number of worker processes
# Add logging settings
loglevel = "info"
errorlog = "logs/gunicorn_error.log"
accesslog = "logs/gunicorn_access.log"