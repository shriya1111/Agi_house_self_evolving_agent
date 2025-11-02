# Self-Evolving Image Generation Agent

An AI system that listens to meetings/voice notes about video creation, extracts video intent and vibe, generates scene frames, evaluates virality through forum scraping, and uses reinforcement learning to optimize image generation prompts.

## Features

- 🎤 **Audio Transcription**: Uses OpenAI Whisper to transcribe meetings and voice notes
- 🧠 **Context Extraction**: GPT-4o analyzes transcriptions to extract video intent, vibe, and requirements
- 🎬 **Scene Generation**: Creates detailed scene breakdowns with start/end frame descriptions
- 🎨 **Image Generation**: Stable Diffusion XL generates high-quality scene frames
- 📊 **Virality Evaluation**: Firecrawl scrapes video forums to extract engagement metrics
- 🤖 **RL Optimization**: Reinforcement learning agent learns to optimize prompts based on virality feedback
- 📈 **Observability**: Weights & Biases tracks all metrics, performance, and resource usage
- 🔧 **Orchestration**: Composio coordinates multi-agent workflow

## Architecture

```
Audio Input → Whisper → GPT-4o Context Extraction → Scene Planning
                                                           ↓
Firecrawl Virality Data ← RL Agent ← Image Generation (Stable Diffusion)
                                   ↓
                         W&B Observability Dashboard
```

## Installation

1. **Clone the repository**:

```bash
git clone <repo-url>
cd Agi_house_self_evolving_agent
```

2. **Create virtual environment**:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:

```bash
pip install -r requirements.txt
```

4. **Set up environment variables**:
   Create a `.env` file in the root directory:

```bash
OPENAI_API_KEY=your_openai_api_key
FIRECRAWL_API_KEY=your_firecrawl_api_key
WANDB_API_KEY=your_wandb_api_key
COMPOSIO_API_KEY=your_composio_api_key  # Optional
```

5. **Configure settings**:
   Edit `config.yaml` to customize model settings, RL parameters, etc.

## Usage

### Basic Usage

```bash
python main.py path/to/audio.mp3 --output-dir outputs
```

### Options

```bash
python main.py audio.mp3 \
  --output-dir outputs \
  --disable-wandb \     # Disable W&B logging
  --disable-rl          # Disable RL optimization
```

### Supported Audio Formats

- MP3
- WAV
- M4A
- FLAC
- Any format supported by Whisper

## Project Structure

```
├── src/
│   ├── audio/              # Audio transcription
│   ├── context/            # Context extraction
│   ├── generation/         # Image generation
│   ├── evaluation/         # Virality evaluation
│   ├── rl/                 # RL prompt optimization
│   ├── observability/      # W&B metrics
│   ├── orchestration/      # Composio orchestration
│   └── config.py           # Configuration management
├── outputs/                # Generated images
├── config.yaml             # Configuration file
├── requirements.txt        # Dependencies
├── main.py                # Main pipeline
└── README.md              # This file
```

## Components

### 1. Audio Transcription (`src/audio/transcriber.py`)

- Uses OpenAI Whisper API or local model
- Supports multiple languages
- Tracks transcription confidence

### 2. Context Extraction (`src/context/extractor.py`)

- GPT-4o extracts video intent, vibe, and style
- Generates structured scene descriptions
- Outputs JSON with start/end frame prompts

### 3. Image Generation (`src/generation/image_generator.py`)

- Stable Diffusion XL for high-quality images
- Generates start and end frames per scene
- Supports GPU acceleration

### 4. Virality Evaluation (`src/evaluation/virality_evaluator.py`)

- Firecrawl scrapes video forums (Reddit, etc.)
- Extracts engagement metrics (upvotes, comments)
- Calculates virality scores

### 5. RL Prompt Optimizer (`src/rl/prompt_optimizer.py`)

- Policy gradient RL agent
- Learns to optimize prompts based on virality rewards
- Updates policy after each generation cycle

### 6. Observability (`src/observability/metrics.py`)

- W&B integration for all metrics
- Tracks transcription quality, image generation, RL rewards
- Monitors GPU/CPU usage

### 7. Orchestration (`src/orchestration/orchestrator.py`)

- Composio coordinates multi-agent workflow
- Manages pipeline execution
- Error handling and recovery

## Configuration

Edit `config.yaml` to customize:

- **Models**: Choose providers (OpenAI/local)
- **RL Parameters**: Learning rate, batch size, exploration rate
- **Firecrawl Targets**: Forum URLs to scrape
- **W&B Settings**: Project name, entity
- **Pipeline Settings**: Enable/disable RL, scene limits

## Requirements

- Python 3.9+
- CUDA-capable GPU (recommended for image generation)
- API keys for:
  - OpenAI (Whisper + GPT-4o)
  - Firecrawl
  - Weights & Biases
  - Composio (optional)

## Troubleshooting

### Image Generation Fails

- Ensure CUDA is installed if using GPU
- Check GPU memory availability
- Try reducing image size in `config.yaml`

### Firecrawl Errors

- Verify `FIRECRAWL_API_KEY` is set
- Check network connectivity
- Review rate limits

### W&B Not Logging

- Verify `WANDB_API_KEY` is set
- Check internet connection
- W&B will fall back to disabled mode if API key is missing

## License

MIT License

## Contributors

Built for the Self-Evolving Agent Build Day hackathon.
