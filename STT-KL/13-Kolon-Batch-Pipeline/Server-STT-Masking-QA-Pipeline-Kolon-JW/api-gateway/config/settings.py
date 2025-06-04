class BaseConfig():
    API_PREFIX = '/api'
    TESTING = False
    DEBUG = False
    UPLOAD_DIR = "/workspace/shared_folder/input_request_folder"
    TA_DIR_DIR = "/workspace/shared_folder/output_stt_ta_folder"
    Maria_DB_HOST = '10.13.8.124'
    Maria_DB_PORT = 3306
    Maria_DB_USER = 'mtm'
    Maria_DB_PW = '@mtm'


class DevConfig(BaseConfig):
    FLASK_ENV = 'development'
    DEBUG = True
    # UPLOAD_DIR = "/home/aithe/01-Wisely/01-System-Microservices/Server-STT-TAs-Dockers/shared_folder/input_request_folder"
    # SQLALCHEMY_DATABASE_URI = 'postgresql://db_user:db_password@db-postgres:5432/flask-deploy'
    # CELERY_BROKER = 'pyamqp://rabbit_user:rabbit_password@broker-rabbitmq//'
    # CELERY_RESULT_BACKEND = 'rpc://rabbit_user:rabbit_password@broker-rabbitmq//'


class ProductionConfig(BaseConfig):
    FLASK_ENV = 'production'
    # SQLALCHEMY_DATABASE_URI = 'postgresql://db_user:db_password@db-postgres:5432/flask-deploy'
    # CELERY_BROKER = 'pyamqp://rabbit_user:rabbit_password@broker-rabbitmq//'
    # CELERY_RESULT_BACKEND = 'rpc://rabbit_user:rabbit_password@broker-rabbitmq//'


class TestConfig(BaseConfig):
    FLASK_ENV = 'development'
    TESTING = True
    DEBUG = True
    # make celery execute tasks synchronously in the same process
    CELERY_ALWAYS_EAGER = True