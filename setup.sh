#!/bin/bash
# Setup script for Self-Evolving Image Generation Agent

echo "Setting up Self-Evolving Image Generation Agent..."
echo ""

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating output directories..."
mkdir -p outputs generated_images logs

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file template..."
    cat > .env << EOF
# OpenAI API Key
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic API Key (optional backup)
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Firecrawl API Key
FIRECRAWL_API_KEY=your_firecrawl_api_key_here

# Weights & Biases API Key
WANDB_API_KEY=your_wandb_api_key_here

# Composio API Key (optional)
COMPOSIO_API_KEY=your_composio_api_key_here

# Model Configuration
IMAGE_MODEL_TYPE=local  # or "replicate" for API
USE_GPU=true
EOF
    echo ".env file created. Please edit it with your API keys."
else
    echo ".env file already exists."
fi

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your API keys"
echo "2. Activate virtual environment: source venv/bin/activate"
echo "3. Run the agent: python main.py path/to/audio.mp3"
echo ""

