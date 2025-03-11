# Tensor.hf_MCP

A HuggingFace integration for Cursor using the Model Context Protocol (MCP).

## Features

- 🔍 **Search Models**: Find models on HuggingFace Hub by name, task, or keywords
- 📚 **Model Information**: Get detailed information about any model
- 💬 **Text Generation**: Generate text using any text generation model
- 🖼️ **Image Generation**: Create images with text-to-image models
- 🔄 **Translation**: Translate text between languages
- 📝 **Summarization**: Summarize long texts
- ❓ **Question Answering**: Answer questions based on context
- 🏷️ **Image Classification**: Classify images using vision models

## Setup

1. Make sure you have a HuggingFace API token
2. Place your token in a file at `docs/Hf_token`
3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage with Cursor

### Installation

To install the HuggingFace MCP server in Cursor:

```bash
mcp install hf_mcp_server.py
```

### Development Mode

For testing and development:

```bash
mcp dev hf_mcp_server.py
```

## Available Tools

- `search_models`: Search for models on HuggingFace Hub
- `text_generation`: Generate text using a text generation model
- `image_generation`: Generate images using a text-to-image model
- `image_classification`: Classify images using a vision model
- `question_answering`: Answer questions based on context
- `summarization`: Summarize text
- `translation`: Translate text between languages
- `get_recommended_models`: Get recommended models for a specific task

## Available Resources

- `hf://models/{model_type}`: List models by type (e.g., text-generation, image-classification)
- `hf://model/{model_id}/info`: Get detailed information about a specific model

## Examples

### Text Generation

```
Use the text_generation tool with model_id="gpt2" and prompt="Once upon a time"
```

### Image Generation

```
Use the image_generation tool with model_id="stabilityai/stable-diffusion-2" and prompt="A beautiful sunset over mountains"
```

### Model Search

```
Use the search_models tool with query="text-to-image" and limit=5
```

## License

MIT