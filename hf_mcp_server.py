from mcp.server.fastmcp import FastMCP, Context, Image
import os
import requests
import json
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import base64
from io import BytesIO
from PIL import Image as PILImage

# Read HuggingFace token from file
def get_hf_token():
    try:
        with open("docs/Hf_token", "r") as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error reading HuggingFace token: {e}")
        return None

# Initialize HuggingFace API token
HF_TOKEN = get_hf_token()
HF_API_BASE = "https://api-inference.huggingface.co/models/"
HF_HUB_API = "https://huggingface.co/api/"

# Create MCP server
mcp = FastMCP("HuggingFace MCP")

# Helper functions for HuggingFace API
def hf_api_headers():
    return {"Authorization": f"Bearer {HF_TOKEN}"}

async def query_model(model_id: str, inputs: Any, parameters: Optional[Dict[str, Any]] = None) -> Any:
    """Query a HuggingFace model with inputs and parameters"""
    payload = {"inputs": inputs}
    if parameters:
        payload["parameters"] = parameters
    
    response = requests.post(
        f"{HF_API_BASE}{model_id}",
        headers=hf_api_headers(),
        json=payload
    )
    
    if response.status_code != 200:
        return f"Error: {response.status_code} - {response.text}"
    
    return response.json()

# Resources
@mcp.resource("hf://models/{model_type}")
def list_models_by_type(model_type: str) -> str:
    """List HuggingFace models by type (text-generation, image-classification, etc.)"""
    response = requests.get(
        f"{HF_HUB_API}models?filter={model_type}&limit=20",
        headers=hf_api_headers()
    )
    
    if response.status_code != 200:
        return f"Error fetching models: {response.status_code} - {response.text}"
    
    models = response.json()
    result = f"# {model_type.capitalize()} Models\n\n"
    for model in models:
        result += f"- **{model['id']}**: {model.get('description', 'No description')[:100]}...\n"
    
    return result

@mcp.resource("hf://model/{model_id}/info")
def get_model_info(model_id: str) -> str:
    """Get detailed information about a specific HuggingFace model"""
    response = requests.get(
        f"{HF_HUB_API}models/{model_id}",
        headers=hf_api_headers()
    )
    
    if response.status_code != 200:
        return f"Error fetching model info: {response.status_code} - {response.text}"
    
    model_info = response.json()
    
    # Format model information
    result = f"# {model_info['id']}\n\n"
    result += f"**Author:** {model_info.get('author', 'Unknown')}\n"
    result += f"**Downloads:** {model_info.get('downloads', 'Unknown')}\n"
    result += f"**Likes:** {model_info.get('likes', 'Unknown')}\n\n"
    
    if model_info.get('cardData'):
        result += f"**Model Card:**\n{model_info['cardData'].get('text', 'No model card')}\n\n"
    
    result += f"**Tags:** {', '.join(model_info.get('tags', []))}\n"
    
    return result

# Tools
@mcp.tool()
async def search_models(query: str, limit: int = 10) -> str:
    """Search for models on HuggingFace Hub"""
    response = requests.get(
        f"{HF_HUB_API}models?search={query}&limit={limit}",
        headers=hf_api_headers()
    )
    
    if response.status_code != 200:
        return f"Error searching models: {response.status_code} - {response.text}"
    
    models = response.json()
    result = f"# Search Results for '{query}'\n\n"
    
    for model in models:
        result += f"- **{model['id']}**: {model.get('description', 'No description')[:100]}...\n"
    
    return result

@mcp.tool()
async def text_generation(model_id: str, prompt: str, max_length: int = 100, temperature: float = 0.7) -> str:
    """Generate text using a HuggingFace text generation model"""
    parameters = {
        "max_length": max_length,
        "temperature": temperature,
        "return_full_text": False
    }
    
    result = await query_model(model_id, prompt, parameters)
    
    if isinstance(result, str) and result.startswith("Error"):
        return result
    
    # Handle different response formats
    if isinstance(result, list) and len(result) > 0:
        if isinstance(result[0], dict) and "generated_text" in result[0]:
            return result[0]["generated_text"]
        elif isinstance(result[0], str):
            return result[0]
    
    return str(result)

@mcp.tool()
async def image_classification(model_id: str, image_url: str) -> str:
    """Classify an image using a HuggingFace image classification model"""
    result = await query_model(model_id, image_url)
    
    if isinstance(result, str) and result.startswith("Error"):
        return result
    
    # Format classification results
    if isinstance(result, list):
        formatted_result = "# Image Classification Results\n\n"
        for item in result:
            if isinstance(item, dict) and "label" in item and "score" in item:
                formatted_result += f"- **{item['label']}**: {item['score']:.4f}\n"
        return formatted_result
    
    return str(result)

@mcp.tool()
async def image_generation(model_id: str, prompt: str) -> Image:
    """Generate an image using a HuggingFace text-to-image model"""
    response = requests.post(
        f"{HF_API_BASE}{model_id}",
        headers=hf_api_headers(),
        json={"inputs": prompt}
    )
    
    if response.status_code != 200:
        raise Exception(f"Error: {response.status_code} - {response.text}")
    
    # Convert image bytes to PIL Image
    image_bytes = response.content
    image = PILImage.open(BytesIO(image_bytes))
    
    # Return as MCP Image
    return Image(data=image_bytes, format=image.format.lower())

@mcp.tool()
async def question_answering(model_id: str, question: str, context: str) -> str:
    """Answer a question based on the provided context using a HuggingFace QA model"""
    payload = {
        "question": question,
        "context": context
    }
    
    result = await query_model(model_id, payload)
    
    if isinstance(result, str) and result.startswith("Error"):
        return result
    
    # Format QA results
    if isinstance(result, dict):
        if "answer" in result:
            return f"Answer: {result['answer']}\nScore: {result.get('score', 'N/A')}"
    
    return str(result)

@mcp.tool()
async def summarization(model_id: str, text: str, max_length: int = 130, min_length: int = 30) -> str:
    """Summarize text using a HuggingFace summarization model"""
    parameters = {
        "max_length": max_length,
        "min_length": min_length
    }
    
    result = await query_model(model_id, text, parameters)
    
    if isinstance(result, str) and result.startswith("Error"):
        return result
    
    # Handle different response formats
    if isinstance(result, list) and len(result) > 0:
        if isinstance(result[0], dict) and "summary_text" in result[0]:
            return result[0]["summary_text"]
    
    return str(result)

@mcp.tool()
async def translation(model_id: str, text: str, source_lang: str = None, target_lang: str = None) -> str:
    """Translate text using a HuggingFace translation model"""
    parameters = {}
    if source_lang:
        parameters["source_lang"] = source_lang
    if target_lang:
        parameters["target_lang"] = target_lang
    
    result = await query_model(model_id, text, parameters)
    
    if isinstance(result, str) and result.startswith("Error"):
        return result
    
    # Handle different response formats
    if isinstance(result, list) and len(result) > 0:
        if isinstance(result[0], dict) and "translation_text" in result[0]:
            return result[0]["translation_text"]
    
    return str(result)

@mcp.tool()
async def get_recommended_models(task: str) -> str:
    """Get recommended models for a specific task"""
    response = requests.get(
        f"{HF_HUB_API}models?filter={task}&sort=downloads&direction=-1&limit=5",
        headers=hf_api_headers()
    )
    
    if response.status_code != 200:
        return f"Error fetching recommended models: {response.status_code} - {response.text}"
    
    models = response.json()
    result = f"# Recommended Models for {task}\n\n"
    
    for model in models:
        result += f"- **{model['id']}**\n"
        result += f"  - Downloads: {model.get('downloads', 'Unknown')}\n"
        result += f"  - Description: {model.get('description', 'No description')[:100]}...\n\n"
    
    return result

# Prompts
@mcp.prompt()
def text_generation_prompt() -> str:
    return """
    I'll help you generate text using HuggingFace models. Please provide:
    
    1. The model ID (e.g., "gpt2", "EleutherAI/gpt-neo-1.3B")
    2. Your prompt text
    3. Optional parameters like max_length and temperature
    
    You can use the search_models tool to find suitable models.
    """

@mcp.prompt()
def image_generation_prompt() -> str:
    return """
    I'll help you generate images using HuggingFace text-to-image models. Please provide:
    
    1. The model ID (e.g., "stabilityai/stable-diffusion-2", "runwayml/stable-diffusion-v1-5")
    2. Your prompt describing the image
    
    You can use the search_models tool with "text-to-image" to find suitable models.
    """

# Run the server
if __name__ == "__main__":
    mcp.run()