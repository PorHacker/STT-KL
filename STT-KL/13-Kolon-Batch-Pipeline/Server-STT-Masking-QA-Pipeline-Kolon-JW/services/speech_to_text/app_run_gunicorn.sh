#!/bin/bash
# gunicorn -c gunicorn_config.py app:app
# 2024/06/25: Replace STT Gunicorn by WSGIServer because of KenLM issue
python app.py
