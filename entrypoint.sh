#!/bin/bash
set -e

# Start the Trading Scheduler in the background
echo "🚀 Starting Trading Scheduler..."
python -m engine.scheduler &

# Capture the PID of the scheduler
SCHEDULER_PID=$!

# Start the Streamlit Dashboard in the foreground
echo "📊 Starting Streamlit Dashboard..."
# Use exec to replace the shell, ensuring signals propagate
# Also binds to 0.0.0.0 for external access
# Use --server.address=0.0.0.0 --server.port=8501 explicitly
exec streamlit run dashboard/app.py --server.address=0.0.0.0 --server.port=8501
