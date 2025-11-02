# API Testing Commands

## Start the API Server

```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies (if not already installed)
pip install fastapi uvicorn python-multipart

# Run the API server
python api.py
```

Or use the helper script:

```bash
./run_api.sh
```

The API will be available at: `http://localhost:8000`

## Test Commands

### 1. Health Check

```bash
curl http://localhost:8000/health
```

### 2. Root Endpoint

```bash
curl http://localhost:8000/
```

### 3. Process Audio File

```bash
curl -X POST http://localhost:8000/api/process-audio \
  -F "audio_file=@test_audio/test_script.mp3" \
  -F "output_dir=outputs" \
  -o response.json
```

Preview the response:

```bash
cat response.json | python -m json.tool | head -50
```

### 4. List Generated Images

```bash
curl http://localhost:8000/api/images
```

### 5. Get Specific Image

```bash
curl http://localhost:8000/api/images/scene_1_optimized.png -o image.png
```

### 6. Run All Tests

```bash
./test_api.sh
```

## Response Format

The `/api/process-audio` endpoint returns JSON with:

```json
{
  "success": true,
  "message": "Processing completed successfully",
  "transcription": "Transcribed text...",
  "context": {
    "intent": "...",
    "vibe": "..."
  },
  "scenes": [
    {
      "scene_number": 1,
      "scene_description": "...",
      "start_frame_prompt": "...",
      "end_frame_prompt": "..."
    }
  ],
  "images": [
    {
      "filename": "scene_1_optimized.png",
      "url": "/images/scene_1_optimized.png",
      "base64": "data:image/png;base64,...",
      "path": "outputs/scene_1_optimized.png"
    }
  ],
  "virality_scores": [...],
  "optimizations": [...]
}
```

## API Documentation

Once the server is running, visit:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
