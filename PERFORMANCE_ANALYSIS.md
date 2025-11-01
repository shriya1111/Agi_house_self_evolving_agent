# Performance Analysis & Optimization Strategy

## Current Pipeline Breakdown

### Step-by-Step Time Estimates

1. **Audio Transcription** (OpenAI Whisper API)

   - Current: ~5-15 seconds for 1-3 minute audio
   - Bottleneck: Network latency + API processing
   - Optimization potential: ✅ Minimal (already fast)

2. **Context Extraction** (GPT-4o API)

   - Current: ~3-5 seconds
   - Bottleneck: API response time
   - Optimization potential: ✅ Minimal (already fast)

3. **Scene Generation** (GPT-4o API)

   - Current: ~3-8 seconds
   - Bottleneck: API response time
   - Optimization potential: ✅ Minimal (already fast)

4. **Image Generation** (Stable Diffusion XL - CPU)

   - Current: **5-15+ minutes PER IMAGE** ⚠️
   - Bottleneck: CPU processing (no GPU acceleration)
   - Optimization potential: 🔴 CRITICAL BOTTLENECK

5. **Virality Evaluation** (Firecrawl API)

   - Current: ~10-30 seconds (multiple forum scrapes)
   - Bottleneck: API rate limits + network
   - Optimization potential: ⚠️ Moderate

6. **RL Optimization** (Local computation)
   - Current: <1 second
   - Bottleneck: None
   - Optimization potential: ✅ Already fast

---

## **Total Current Time**

- **Without images**: ~20-50 seconds
- **With images (5 scenes × 2 frames each)**: **50-150+ MINUTES** ⚠️

---

## Target: 10 Seconds End-to-End

### Realistic Options:

#### Option 1: **Skip Image Generation** ✅ FASTEST

- Time: ~20-30 seconds
- What you get: Transcription, context, scenes, prompts ready
- Use case: Preview/validation workflow

#### Option 2: **Limit to 1 Image** ⚠️ MODERATE

- Time: ~25-45 seconds
- Generate: Only 1 representative scene frame
- Use case: Quick preview of one scene

#### Option 3: **Use API-Based Image Generation** 🎯 RECOMMENDED

- Time: ~15-30 seconds total
- Services: Replicate API, Stability AI API, or OpenAI DALL-E
- Trade-off: API costs vs speed
- Use case: Production-ready speed

#### Option 4: **GPU Acceleration** 🚀 BEST QUALITY

- Time: ~10-20 seconds per image (with GPU)
- Requirements: CUDA-capable GPU (NVIDIA)
- Use case: Best quality + speed if GPU available

---

## Optimization Recommendations (Priority Order)

### 🔴 CRITICAL: Image Generation

**Current Problem**: CPU-based Stable Diffusion XL is extremely slow (5-15 min/image)

**Solutions (choose one)**:

1. **Use Replicate API** (RECOMMENDED for 10-second target)

   - Fast: ~3-5 seconds per image
   - Cost: ~$0.003 per image
   - Quality: High
   - No local setup needed

2. **Use Stability AI API**

   - Fast: ~3-5 seconds per image
   - Cost: Similar to Replicate
   - Quality: High

3. **Use OpenAI DALL-E 3**

   - Fast: ~10-15 seconds per image
   - Cost: ~$0.04-0.08 per image
   - Quality: Excellent
   - Easiest integration

4. **Limit Image Count**

   - Generate only 1 scene (2 frames) instead of 5 scenes
   - Time saved: ~40-60 minutes
   - Trade-off: Less preview content

5. **Skip Image Generation (Preview Mode)**
   - Generate prompts but not images
   - User can generate images later
   - Time: <30 seconds total

### ⚠️ MODERATE: Virality Evaluation

**Current Problem**: Scraping multiple forums takes 10-30 seconds

**Solutions**:

1. **Parallel scraping** (if API supports)
2. **Cache results** for same keywords
3. **Limit to 1-2 forums** instead of 3+
4. **Make optional** - skip in fast mode

### ✅ MINOR: API Calls

**Current**: Already optimized

- Transcription: Fast enough
- GPT-4o: Fast enough
- Consider: Batch requests if possible

---

## Recommended Configuration for 10-Second Target

### Fast Mode Configuration:

```yaml
# Fast mode (10-30 seconds)
pipeline:
  fast_mode: true
  skip_image_generation: false # Set to true for <10 seconds
  limit_images: 1 # Generate only 1 scene
  virality_evaluation: false # Skip virality checks
  use_api_images: true # Use Replicate/Stability AI API

image_generation:
  provider: "replicate" # or "stability-ai", "dall-e"
  model: "stability-ai/sdxl" # Fast API model
  limit_scenes: 1 # Only first scene
```

### Implementation Strategy:

1. **Phase 1: Skip Images** (Instant - <30 seconds)

   - Generate all prompts and scenes
   - Show prompts ready for generation
   - User can generate images separately

2. **Phase 2: Single Preview Image** (~15-30 seconds)

   - Use Replicate/Stability API for 1 quick preview
   - Fast enough for user feedback

3. **Phase 3: Full Generation** (Background task)
   - Generate all images asynchronously
   - User gets preview first, full generation later

---

## Realistic Targets

| Target          | Configuration              | Time       | Quality      |
| --------------- | -------------------------- | ---------- | ------------ |
| **<10 seconds** | Skip images, skip virality | 8-15 sec   | Prompts only |
| **<30 seconds** | 1 API image, skip virality | 20-35 sec  | 1 preview    |
| **<2 minutes**  | All scenes via API         | 60-120 sec | Full preview |
| **<10 minutes** | GPU + optimized            | 5-8 min    | Full quality |

---

## Recommendation for Your Use Case

**For 10-second target**, I recommend:

1. **Skip image generation in main pipeline** ✅

   - Generate all prompts and scenes (<30 sec)
   - Store prompts for later generation
   - User gets immediate feedback

2. **Add separate "Generate Images" command** ✅

   - `python generate_images.py --from-latest`
   - Runs asynchronously
   - Uses Replicate API for speed

3. **Make virality evaluation optional** ✅
   - Only run when explicitly requested
   - Or run asynchronously

**Result**: Main pipeline <30 seconds, image generation separate (async)

---

## Code Changes Needed (Summary)

1. Add `fast_mode` flag to config
2. Make image generation optional
3. Integrate Replicate/Stability API (or DALL-E)
4. Add async image generation option
5. Make virality evaluation optional/skippable
