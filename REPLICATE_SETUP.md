# Replicate API Setup

## Quick Setup

1. **Get your Replicate API token**:

   - Go to https://replicate.com/account/api-tokens
   - Sign up or log in
   - Create a new API token
   - Copy the token

2. **Add to .env file**:

   ```bash
   REPLICATE_API_TOKEN=your_replicate_api_token_here
   ```

3. **That's it!** The system will use Replicate API for fast image generation.

## Cost

- ~$0.003 per image (very affordable)
- Fast: ~3-5 seconds per image
- No GPU needed on your machine

## Preview Mode

The system is configured to generate **only 1 preview image** by default (first scene, start frame).

This gives you:

- **Fast feedback** (20-30 seconds total)
- **Preview of quality** before generating all images
- **Cost efficient** (~$0.003 per run)

## Changing Settings

Edit `config.yaml`:

```yaml
pipeline:
  preview_mode: true # Set to false for full generation
  limit_preview_images: 1 # Number of preview images

image_generation:
  provider: "replicate" # Fast API-based generation
  num_inference_steps: 25 # Lower = faster (quality still good)
```

## Full Generation

To generate all scenes (not just preview):

1. Set `preview_mode: false` in `config.yaml`, OR
2. Run with `--full-generation` flag (if implemented)

This will generate all scenes with both start and end frames (slower, more expensive, but complete).
