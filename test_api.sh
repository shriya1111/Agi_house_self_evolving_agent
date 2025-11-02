#!/bin/bash
# Test script for the API

API_URL="http://localhost:8000"
AUDIO_FILE="test_audio/test_script.mp3"

echo "=========================================="
echo "Testing Self-Evolving Image Generation API"
echo "=========================================="
echo ""

# Test 1: Health check
echo "1. Testing health endpoint..."
curl -s "$API_URL/health" | python -m json.tool
echo ""
echo ""

# Test 2: Root endpoint
echo "2. Testing root endpoint..."
curl -s "$API_URL/" | python -m json.tool
echo ""
echo ""

# Test 3: Process audio file
if [ -f "$AUDIO_FILE" ]; then
    echo "3. Testing audio processing (this will take 30-60 seconds)..."
    echo "   Uploading: $AUDIO_FILE"
    echo ""
    curl -X POST "$API_URL/api/process-audio" \
        -F "audio_file=@$AUDIO_FILE" \
        -F "output_dir=outputs" \
        -o response.json
    
    echo ""
    echo "Response saved to response.json"
    echo "Preview:"
    python -c "import json; data = json.load(open('response.json')); print(f\"Success: {data.get('success')}\"); print(f\"Images: {len(data.get('images', []))} generated\"); print(f\"Scenes: {len(data.get('scenes', []))} generated\")"
    echo ""
else
    echo "3. Skipping audio test - $AUDIO_FILE not found"
    echo ""
fi

# Test 4: List images
echo "4. Testing list images endpoint..."
curl -s "$API_URL/api/images" | python -m json.tool
echo ""
echo ""

echo "=========================================="
echo "Tests complete!"
echo "=========================================="

