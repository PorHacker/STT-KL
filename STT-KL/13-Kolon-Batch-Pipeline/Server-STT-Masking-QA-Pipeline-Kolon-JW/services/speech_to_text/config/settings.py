class BaseConfig():
    API_PREFIX = '/api'
    TESTING = False
    DEBUG = False
    JSON_AS_ASCII=False
    # STT_MODEL_CHECKPOINT = "/workspace/models/UNICEF-Finetune-Base-Model.nemo"  # CTC model Checkpoint
    # STT_MODEL_CHECKPOINT = "/workspace/models/Conformer-CTC-BPE-Large-Pretrain-Consult-Unigram-Combined-Dataset-New-TOK.nemo"  # AITHE BASE model Checkpoint
    # KENLM_MODEL_CHECKPOINT = "/workspace/models/kenlm.bin"

    # KOLON MODELS
    STT_MODEL_CHECKPOINT = "/workspace/models/Conformer-CTC-BPE-Large-Pretrain-Consult-Unigram-KOLON-Finetune-From-Base-LR01.nemo"  # AITHE BASE model Checkpoint
    KENLM_MODEL_CHECKPOINT = "/workspace/models/kenlm_kolon.bin"


class DevConfig(BaseConfig):
    FLASK_ENV = 'development'
    DEBUG = True
    # STT_MODEL_CHECKPOINT = "/home/aithe/01-Wisely/01-System-Microservices/Server-STT-TAs-Dockers/services/speech_to_text/models/STT-METAM-Finetune-From-Base.nemo"  # CTC model Checkpoint
    # KENLM_MODEL_CHECKPOINT = "/home/aithe/01-Wisely/01-System-Microservices/Server-STT-TAs-Dockers/services/speech_to_text/models/kenlm.bin"
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
    # CELERY_ALWAYS_EAGER = True