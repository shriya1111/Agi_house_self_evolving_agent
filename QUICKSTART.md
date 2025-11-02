# Quick Start Guide

## Prerequisites

1. Python 3.9 or higher
2. CUDA-capable GPU (recommended for image generation)
3. API keys:
   - OpenAI API key
   - Firecrawl API key
   - Weights & Biases API key
   - Composio API key (optional)

## Setup (5 minutes)

```bash
# 1. Run setup script
./setup.sh

# 2. Activate virtual environment
source venv/bin/activate

# 3. Edit .env file with your API keys
nano .env  # or use your favorite editor
```

## First Run

```bash
# Process an audio file
python main.py path/to/your/audio.mp3 --output-dir outputs
```

## Example Workflow

1. **Record a voice note** describing the video you want to create

   - Example: "I want to make a fast-paced tech product demo video with energetic music, bright colors, and dynamic camera movements. The video should showcase the product's key features in an exciting way."

2. **Save as audio file** (MP3, WAV, etc.)

3. **Run the agent**:

   ```bash
   python main.py your_voice_note.mp3
   ```

4. **Check outputs**:
   - Generated images in `outputs/` directory
   - W&B dashboard for metrics
   - Console output for workflow progress

## Configuration

Edit `config.yaml` to customize:

- Model providers (OpenAI/local)
- RL learning rate and parameters
- Firecrawl target forums
- Image generation settings

## Troubleshooting

### "Model not loaded" error

- Check if CUDA is installed (for GPU)
- Try reducing image size in `config.yaml`
- Use CPU mode if GPU unavailable

### "API key not found" errors

- Verify `.env` file has correct API keys
- Ensure `.env` is in project root directory

### Firecrawl scraping fails

- Check `FIRECRAWL_API_KEY` is set
- Verify network connectivity
- Review rate limits

## Advanced Usage

```bash
# Disable W&B logging
python main.py audio.mp3 --disable-wandb

# Disable RL optimization
python main.py audio.mp3 --disable-rl

# Custom output directory
python main.py audio.mp3 --output-dir my_outputs
```

## Testing Imports

```bash
python test_imports.py
```

This verifies all modules can be imported correctly.

## Next Steps

- Check `README.md` for full documentation
- Explore `config.yaml` for customization
- Review W&B dashboard for metrics
- Experiment with different audio inputs
