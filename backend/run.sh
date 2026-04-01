#!/bin/bash
set -e
cd backend
../.venv/bin/python -m pip install -r requirements.txt
../.venv/bin/python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
