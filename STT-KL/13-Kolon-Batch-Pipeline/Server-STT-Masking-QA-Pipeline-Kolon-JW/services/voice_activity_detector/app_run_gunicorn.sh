#!/bin/bash
unset LD_LIBRARY_PATH
gunicorn -c gunicorn_config.py app:app