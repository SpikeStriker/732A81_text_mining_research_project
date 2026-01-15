#!/bin/bash

# Start Qdrant in background
echo "Starting Qdrant..."
qdrant --config-path /workspace/qdrant_config.yaml &
sleep 3

# Start LLM server in background
echo "Starting LLM server..."
python llm_server.py &

# Wait for services to start
sleep 5

echo "Checking services..."
curl -s http://localhost:6333 > /dev/null && echo "✓ Qdrant is running" || echo "✗ Qdrant failed"
curl -s http://localhost:8001/health > /dev/null && echo "✓ LLM server is running" || echo "✗ LLM server failed"


# Start Jupyter notebook (main process)
echo "Starting Jupyter notebook..."
jupyter notebook \
    --ip=0.0.0.0 \
    --port=8888 \
    --no-browser \
    --allow-root \
    --NotebookApp.token='whatever123' \
    --NotebookApp.allow_origin='*' \
    --NotebookApp.disable_check_xsrf=True