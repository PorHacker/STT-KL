# gunicorn_config.py
workers = 32
worker_class = "uvicorn.workers.UvicornWorker"
bind = "0.0.0.0:9889"
timeout = 600
# Add logging settings
loglevel = "debug"
errorlog = "logs/gunicorn_error.log"
accesslog = "logs/gunicorn_access.log"