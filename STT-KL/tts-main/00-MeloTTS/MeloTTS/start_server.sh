#!/bin/bash
export PYTHONWARNINGS="ignore"
gunicorn -c gunicorn_config.py tts_api:APP 