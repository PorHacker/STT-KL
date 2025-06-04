#!/bin/bash
# export CUDA_VISIBLE_DEVICES=-1
gunicorn -c gunicorn_config.py app:app
