"""
test_client.py

This file contains a suite of tests that exercise the features of the GeminiClient
and APIWrapper as defined in client.py. In addition to testing text generation, 
image inputs, streaming, caching, and tool enabling, it now includes a test that 
ensures each API key found in the environment is used at least once.

A dummy client is patched in place of the actual google.genai.Client to simulate
API responses without making external calls. The dummy client records every API 
key used in a global set for verification.

Usage:
    python test_client.py
"""

import asyncio
import unittest
import tempfile
import os
import time
import PIL.Image
from fastapi.testclient import TestClient

# Import the GeminiClient and APIWrapper from client.py
from client import GeminiClient, APIWrapper

# Patch the google.genai.Client with a dummy implementation
from google import genai

# Global set to track which API keys are used during testing.
USED_API_KEYS = set()

class DummyResponse:
    """A dummy response object to simulate API responses."""
    def __init__(self, text):
        self.text = text
        self.latency = 0.1

    def __str__(self):
        return self.text

class DummyModels:
    """Simulates the models interface for the dummy client."""
    async def generate_content(self, model, contents, config):
        # For list inputs (text and/or images), join their string representations.
        if isinstance(contents, list):
            joined = " | ".join([str(c) for c in contents])
        else:
            joined = str(contents)
        return DummyResponse(text=f"Dummy response to: {joined}")

    async def generate_content_stream(self, model, contents, config):
        # Simulate a streaming response by yielding DummyResponse objects.
        for chunk in ["Dummy ", "streamed ", "response."]:
            await asyncio.sleep(0.01)
            yield DummyResponse(text=chunk)

class DummyClient:
    """A dummy client that replaces the google.genai.Client for testing purposes.
    
    It records the API key used for each instance in the global USED_API_KEYS set.
    """
    def __init__(self, api_key):
        self.api_key = api_key
        USED_API_KEYS.add(api_key)
        self.models = DummyModels()

# Save the original client to restore after tests.
original_genai_client = genai.Client
genai.Client = DummyClient

class TestGeminiClient(unittest.IsolatedAsyncioTestCase):
    """Test cases for GeminiClient features using asynchronous tests."""
    async def asyncSetUp(self):
        # Set up dummy API keys in the environment (keys must start with "AIzaSy")
        os.environ['GEMINI_API_KEY1'] = "AIzaSy_dummy1"
        os.environ['GEMINI_API_KEY2'] = "AIzaSy_dummy2"
        os.environ['GEMINI_API_KEY3'] = "AIzaSy_dummy3"
        os.environ['GEMINI_API_KEY4'] = "AIzaSy_dummy4"
        # Create a GeminiClient instance with a temporary cache directory.
        # Set key_rotation_interval to a very small value to force rotation.
        self.client = GeminiClient(
            key_rotation_interval=0.001,
            improve_prompts=True,
            cache_dir=tempfile.mkdtemp()
        )
    
    async def test_text_generation(self):
        """Test that a simple text prompt returns the dummy response."""
        prompt = "How does AI work?"
        response = await self.client.generate_content_async(prompt)
        self.assertIn("Dummy response to:", response)
        # Verify that conversation history is updated.
        self.assertTrue(len(self.client.conversation_history) > 0)

    async def test_image_generation(self):
        """Test that an image input along with a text prompt returns a valid response."""
        # Create a dummy red image using PIL.
        image = PIL.Image.new('RGB', (100, 100), color='red')
        prompt = "What is this image?"
        response = await self.client.generate_content_async([prompt, image])
        self.assertIn("Dummy response to:", response)

    async def test_generate_stream(self):
        """Test that streaming response generation yields chunks of text."""
        prompt = "Tell a story"
        stream_gen = self.client.generate_content_stream(prompt)
        chunks = []
        # Iterate asynchronously over the stream generator
        async for chunk in stream_gen:
            # The dummy may yield either DummyResponse objects or plain strings.
            if hasattr(chunk, "text"):
                chunks.append(chunk.text)
            else:
                chunks.append(chunk)
        full_response = "".join(chunks)
        self.assertIn("Dummy", full_response)

    async def test_caching(self):
        """Test that repeated prompts return the cached response when conversation history is identical."""
        prompt = "Explain quantum computing"
        # First call: conversation history is empty.
        response1 = await self.client.generate_content_async(prompt)
        # Clear conversation history so that the prompt remains identical.
        self.client.conversation_history.clear()
        response2 = await self.client.generate_content_async(prompt)
        self.assertEqual(response1, response2)

    async def test_enable_tools(self):
        """Test enabling additional tools in the generation configuration."""
        self.client.enable_code_execution()
        self.client.enable_google_search()
        self.client.enable_retrieval()
        # Enable function calling with a dummy function declaration.
        dummy_function = {"name": "dummy", "description": "A dummy function"}
        self.client.enable_function_calling([dummy_function])
        config = self.client.generation_config
        self.assertIn("tools", config)
        self.assertIn("function_declarations", config)

    async def test_improve_prompt(self):
        """Test that the improve_prompt method returns a non-empty string."""
        prompt = "Short prompt"
        improved = await self.client.improve_prompt(prompt)
        self.assertIsInstance(improved, str)
        self.assertNotEqual(improved.strip(), "")

    async def test_api_key_rotation(self):
        """
        Test that each API key in the environment is used at least once.
        
        This test makes a series of unique requests so that caching does not interfere.
        With a very small key rotation interval, each call should rotate to a new key.
        """
        # Clear the global set for this test.
        global USED_API_KEYS
        USED_API_KEYS.clear()
        
        # Retrieve the expected keys from the environment.
        expected_keys = {
            os.environ['GEMINI_API_KEY1'],
            os.environ['GEMINI_API_KEY2'],
            os.environ['GEMINI_API_KEY3'],
            os.environ['GEMINI_API_KEY4']
        }
        # Make as many unique requests as there are expected keys.
        for i in range(len(expected_keys)):
            prompt = f"Unique prompt {i} at {time.time()}"
            # Ensure unique prompt to avoid cache hits.
            await self.client.generate_content_async(prompt)
            # Wait a tiny bit to force key rotation.
            await asyncio.sleep(0.005)
        
        # Check that every expected key was used at least once.
        self.assertEqual(USED_API_KEYS, expected_keys)

class TestAPIWrapper(unittest.TestCase):
    """Test cases for the FastAPI endpoints provided by APIWrapper."""
    def setUp(self):
        # Ensure dummy API keys are present.
        os.environ['GEMINI_API_KEY1'] = "AIzaSy_dummy1"
        os.environ['GEMINI_API_KEY2'] = "AIzaSy_dummy2"
        os.environ['GEMINI_API_KEY3'] = "AIzaSy_dummy3"
        os.environ['GEMINI_API_KEY4'] = "AIzaSy_dummy4"
        # Create a GeminiClient instance.
        self.client = GeminiClient(
            key_rotation_interval=1,
            improve_prompts=False,
            cache_dir=tempfile.mkdtemp()
        )
        self.api_wrapper = APIWrapper(self.client)
        self.test_client = TestClient(self.api_wrapper.app)

    def test_generate_endpoint(self):
        """Test the /generate endpoint for text generation."""
        response = self.test_client.post("/generate", json={"prompt": "What is AI?"})
        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        self.assertIn("response", json_data)
        self.assertIn("Dummy response to:", json_data["response"])

    def test_websocket_endpoint(self):
        """Test the WebSocket endpoint for real-time interactions."""
        with self.test_client.websocket_connect("/ws") as websocket:
            websocket.send_text("Tell me a joke")
            data = websocket.receive_text()
            self.assertIn("Dummy response to:", data)

if __name__ == '__main__':
    unittest.main()

# Restore the original genai.Client after the tests have finished.
genai.Client = original_genai_client
