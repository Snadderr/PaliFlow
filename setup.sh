#!/bin/bash
set -e

echo "=== PaliFlow Study Setup ==="

# System dependencies
apt-get update && apt-get install -y tesseract-ocr

# Python dependencies
pip install -r requirements.txt

# HuggingFace login (needed for Gemma 2 27B license)
echo ""
echo "If you haven't already, run: huggingface-cli login"
echo "Gemma 2 27B requires accepting the license at https://huggingface.co/google/gemma-2-27b-it"

# Create directories
mkdir -p cache results

echo ""
echo "Setup complete. Next steps:"
echo "  1. Set ROBOFLOW_API_KEY:  export ROBOFLOW_API_KEY=your_key_here"
echo "  2. Run preprocessing:     python preprocess.py"
echo "  3. Start the study GUI:   streamlit run app.py --server.port 8501 --server.address 0.0.0.0"
