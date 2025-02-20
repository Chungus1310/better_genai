"""
test_client.py

This script demonstrates the functionality of the GeminiClient class, including:
- API key rotation and secure storage
- Generating text responses from Gemini
- Handling image-based prompts
- Managing conversation history
- Streaming responses
- Using the FastAPI wrapper for HTTP/WebSocket requests

Make sure `client.py` is in the same directory and API keys are set in the `.env` file.
"""

import asyncio
import logging
import PIL.Image
from client import GeminiClient, APIWrapper
import uvicorn
import threading
import requests
import json
import websockets

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the GeminiClient
gemini_client = GeminiClient(
    key_rotation_interval=0.5,  # Ensure rapid key rotation
    improve_prompts=True,
    cache_dir="./test_cache"
)

async def test_text_generation():
    """Tests basic text generation using Gemini."""
    logger.info("Testing basic text generation...")
    response = await gemini_client.generate_content_async("Explain quantum computing in simple terms.")
    print(f"Response:\n{response}\n")

async def test_conversation_history():
    """Tests conversation memory by sending sequential prompts."""
    logger.info("Testing conversation history...")
    
    await gemini_client.generate_content_async("Hello! Who are you?")
    response = await gemini_client.generate_content_async("What did I just ask you?")
    
    print(f"Conversation Memory Response:\n{response}\n")

async def test_image_prompt():
    """Tests sending an image as part of the request."""
    logger.info("Testing image-based query...")
    
    image = PIL.Image.open("test.png")  # Replace with a real image path
    
    response = await gemini_client.generate_content_async(["What is this image?", image])
    print(f"Image Response:\n{response}\n")

async def test_streaming_response():
    """Tests streaming a long response from Gemini."""
    logger.info("Testing streaming response...")
    
    print("Streaming Response:\n")
    async for chunk in gemini_client.generate_content_stream("Tell me a long story about space exploration"):
        print(chunk, end="", flush=True)
    print("\n")

async def test_batch_generation():
    """Tests sending multiple prompts in a batch."""
    logger.info("Testing batch text generation...")
    
    prompts = [
        "Explain relativity in simple terms.",
        "Summarize the history of AI.",
        "Describe the future of robotics."
    ]
    
    responses = await gemini_client.generate_batch(prompts)
    for i, response in enumerate(responses):
        print(f"Response {i+1}:\n{response}\n")

async def test_prompt_improvement():
    """Tests the prompt improvement functionality."""
    logger.info("Testing prompt improvement...")
    
    # Unoptimized prompt
    unoptimized_prompt = "tell me what deep learning is and give examples"
    logger.info(f"\nUnoptimized prompt:\n{unoptimized_prompt}")
    
    # Get improved prompt
    improved_prompt = await gemini_client.improve_prompt(unoptimized_prompt)
    logger.info(f"\nImproved prompt:\n{improved_prompt}")
    
    # Generate responses for both prompts for comparison
    unopt_response = await gemini_client.generate_content_async(unoptimized_prompt)
    opt_response = await gemini_client.generate_content_async(improved_prompt)
    
    print("\nUnoptimized Response:")
    print("-" * 40)
    print(unopt_response)
    print("\nOptimized Response:")
    print("-" * 40)
    print(opt_response)

def start_fastapi_server():
    """Starts the FastAPI server in a separate thread."""
    api_wrapper = APIWrapper(gemini_client)
    server_thread = threading.Thread(target=lambda: uvicorn.run(api_wrapper.app, host="0.0.0.0", port=8000))
    server_thread.daemon = True
    server_thread.start()

def test_fastapi_http_request():
    """Tests the FastAPI HTTP endpoint for text generation."""
    logger.info("Testing FastAPI HTTP request...")

    url = "http://127.0.0.1:8000/generate"
    data = {"prompt": "Tell me a fun fact about space."}
    
    response = requests.post(url, json=data)
    if response.status_code == 200:
        print(f"FastAPI HTTP Response:\n{response.json()['response']}\n")
    else:
        print("FastAPI HTTP request failed.")

async def test_fastapi_websocket():
    """Tests the FastAPI WebSocket endpoint for real-time responses."""
    logger.info("Testing FastAPI WebSocket connection...")

    uri = "ws://127.0.0.1:8000/ws"
    async with websockets.connect(uri) as websocket:
        await websocket.send("Tell me a joke.")
        response = await websocket.recv()
        print(f"WebSocket Response:\n{response}\n")

async def run_all_tests():
    """Runs all test functions sequentially."""
    await test_text_generation()
    await test_conversation_history()
    await test_image_prompt()
    await test_streaming_response()
    await test_batch_generation()
    await test_prompt_improvement() 

if __name__ == "__main__":
    # Start FastAPI server
    api_wrapper = APIWrapper(gemini_client)
    server = start_fastapi_server()

    # Run all async tests
    asyncio.run(run_all_tests())

    # Run FastAPI HTTP and WebSocket tests
    test_fastapi_http_request()
    asyncio.run(test_fastapi_websocket())

    server.should_exit = True  

