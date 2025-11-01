# Quick Switch Guide - Image Generation Options

## Current Default: **"skip" mode** (FREE, No Images)

The system is now set to **skip image generation** by default.

**Result:**

- ✅ Time: ~20 seconds total
- ✅ Cost: FREE (only OpenAI API)
- ✅ Output: All prompts and scenes ready

---

## Switch to DALL-E (If You Want Images)

### Option 1: Use DALL-E 3 (Uses Your OpenAI Key) ✅

**Edit `config.yaml`:**

```yaml
image_generation:
  provider: "dall-e" # Change from "skip" to "dall-e"
```

**Result:**

- Time: ~30-40 seconds total (10-15 sec per image)
- Cost: ~$0.04-0.08 per image
- Quality: Excellent
- Uses: Your existing OpenAI API key (no new account needed!)

---

## Other Options

### Option 2: Skip Images (Current Default) ✅ FREE

```yaml
provider: "skip"
```

- Fastest
- Free
- Prompts ready for later generation

### Option 3: Local GPU (If Available) ✅ FREE

```yaml
provider: "local"
```

- Requires NVIDIA GPU
- Free (your hardware)
- ~10-20 seconds per image

### Option 4: Replicate (If You Get It Working)

```yaml
provider: "replicate"
```

- Fast: ~3-5 seconds per image
- Cost: ~$0.003 per image
- Requires: Replicate API token

---

## Recommendation

**For Hackathon Demo:**

1. **Use "skip" mode** (current default) - FREE, fast, shows all prompts
2. **OR switch to "dall-e"** if you want 1 preview image - uses your OpenAI key

**Edit `config.yaml` to switch between options!**
