# gunicorn_config.py
bind = "0.0.0.0:5004"  # Bind the server to this address and port
timeout  = 86400
# Add logging settings
loglevel = "info"
errorlog = "logs/gunicorn_error.log"
accesslog = "logs/gunicorn_access.log"