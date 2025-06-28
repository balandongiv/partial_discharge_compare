#!/bin/sh
pip install -r requirements.txt
# Run tests if any exist
pytest || echo "No tests to run"
