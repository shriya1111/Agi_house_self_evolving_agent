# Setting Up API Keys

## Quick Setup

1. **Create a `.env` file** in the project root directory:

   ```bash
   cp .env.example .env
   ```

2. **Edit the `.env` file** and replace all `your_*_api_key_here` placeholders with your actual API keys.

## Required API Keys

### 1. OpenAI API Key (Required)

- **Where to get it**: https://platform.openai.com/api-keys
- **Required for**:
  - Audio transcription (Whisper)
  - Context extraction (GPT-4o)
  - Scene generation (GPT-4o)
- **Example**:
  ```bash
  OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
  ```

### 2. Firecrawl API Key (Required)

- **Where to get it**: https://firecrawl.dev/app/settings?tab=billing
- **Free credits**: Use code `agihouse10k` for 10,000 free credits
- **Required for**: Virality evaluation (scraping video forums)
- **Example**:
  ```bash
  FIRECRAWL_API_KEY=fc-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
  ```

### 3. Weights & Biases API Key (Required)

- **Where to get it**: https://wandb.ai/settings
- **Free credits**: Fill out https://wandb.me/whform for $50 free inference credits
- **Required for**: Observability and metrics tracking
- **Example**:
  ```bash
  WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
  ```

## Optional API Keys

### 4. Anthropic API Key (Optional)

- **Where to get it**: https://console.anthropic.com/
- **Use case**: Backup LLM provider (not currently used, but available)
- **Example**:
  ```bash
  ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
  ```

### 5. Composio API Key (Optional)

- **Where to get it**: https://app.composio.dev/
- **Use case**: Advanced orchestration features
- **Example**:
  ```bash
  COMPOSIO_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
  ```

## File Location

The `.env` file should be in the **root directory** of the project:

```
Agi_house_self_evolving_agent/
├── .env              <-- Put your API keys here
├── .env.example      <-- Template file
├── config.yaml
├── main.py
└── src/
```

## Verification

After setting up your API keys, test the configuration:

```bash
# Activate virtual environment
source venv/bin/activate

# Run import test
python test_imports.py
```

## Security Notes

⚠️ **Important**:

- Never commit your `.env` file to git (it's already in `.gitignore`)
- Keep your API keys secure and private
- Rotate keys if they're exposed
- Use environment-specific keys for production

## Getting API Keys

### OpenAI

1. Go to https://platform.openai.com/
2. Sign up or log in
3. Navigate to API Keys section
4. Click "Create new secret key"
5. Copy the key (you won't see it again!)

### Firecrawl

1. Go to https://firecrawl.dev/app/settings?tab=billing
2. Sign up or log in
3. Apply coupon code: `agihouse10k`
4. Copy your API key from settings

### Weights & Biases

1. Go to https://wandb.ai/
2. Sign up or log in
3. Go to Settings → API Keys
4. Copy your API key
5. Also fill out: https://wandb.me/whform for free credits

## Example .env File

```bash
# Required
OPENAI_API_KEY=sk-proj-abc123def456ghi789jkl012mno345pqr678stu901vwx234yz
FIRECRAWL_API_KEY=fc-abc123def456ghi789jkl012mno345pqr678stu901vwx234yz
WANDB_API_KEY=abc123def456ghi789jkl012mno345pqr678stu901vwx234yzabc123def456

# Optional
ANTHROPIC_API_KEY=sk-ant-abc123def456ghi789jkl012mno345pqr678stu901vwx234yz
COMPOSIO_API_KEY=abc123def456ghi789jkl012mno345pqr678stu901vwx234yz
```
