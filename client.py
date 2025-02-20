"""
client.py

Revised client file using the new google‑genai module (the successor to google‑generativeai).
It loads API keys from a local .env file (keys must start with GEMINI_API_KEY),
stores them in encrypted form, rotates between multiple API keys (one per request with a timeout
of 0.5 seconds) to mitigate rate limits, and maintains conversation history (up to 4 messages)
that is sent with every request.
"""

import asyncio
import os
import time
import hashlib
import json
import base64
from collections import deque
from typing import Optional, Dict, Any, List, Generator, Callable, Union

from google import genai
from google.genai import types

from diskcache import Cache
from prometheus_client import Histogram, Counter
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
import uvicorn
from pydantic import BaseModel
from dotenv import load_dotenv
import PIL.Image

# Load environment variables from .env file
load_dotenv()

# Basic logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#################################
# Simple Encryption/Decryption Helpers
#################################

def xor_encrypt(data: bytes, key: bytes) -> bytes:
    """Encrypts/Decrypts data using a repeating XOR cipher."""
    return bytes([b ^ key[i % len(key)] for i, b in enumerate(data)])

def encrypt_key(key: str, secret: str) -> str:
    """Encrypts the given API key using XOR and returns a base64 encoded string."""
    key_bytes = key.encode('utf-8')
    secret_bytes = secret.encode('utf-8')
    encrypted_bytes = xor_encrypt(key_bytes, secret_bytes)
    return base64.urlsafe_b64encode(encrypted_bytes).decode('utf-8')

def decrypt_key(enc_key: str, secret: str) -> str:
    """Decrypts the given encrypted key (base64 encoded) using XOR and returns the original key."""
    encrypted_bytes = base64.urlsafe_b64decode(enc_key.encode('utf-8'))
    secret_bytes = secret.encode('utf-8')
    decrypted_bytes = xor_encrypt(encrypted_bytes, secret_bytes)
    return decrypted_bytes.decode('utf-8')

#################################
# Utility and Helper Classes
#################################

class TokenBucketRateLimiter:
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.time()

    def consume(self) -> bool:
        self._refill()
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False

    def _refill(self):
        current_time = time.time()
        elapsed = current_time - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = current_time

class KeyManager:
    def __init__(self, keys: List[str], capacity: int, refill_rate: float, secret: str):
        self.secret = secret
        # Store keys in encrypted form.
        self.keys = [
            {
                "encrypted_key": key,
                "rate_limiter": TokenBucketRateLimiter(capacity, refill_rate),
                "usage": 0,
                "last_used": 0,
            }
            for key in keys
        ]
        self.current_index = 0  # Track current key index

    def get_next_key(self) -> str:
        start_index = self.current_index
        while True:
            key_info = self.keys[self.current_index]
            # Rotate index for next request
            self.current_index = (self.current_index + 1) % len(self.keys)
            if key_info["rate_limiter"].consume():
                key_info["last_used"] = time.time()
                key_info["usage"] += 1
                # Decrypt and return the key
                return decrypt_key(key_info["encrypted_key"], self.secret)
            if self.current_index == start_index:
                raise Exception("All API keys are rate limited")

class MetricsCollector:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.request_latency = Histogram("gemini_request_latency", "Request latency in seconds")
            cls._instance.tokens_used = Counter("gemini_tokens_used", "Total tokens used")
            cls._instance.errors = Counter("gemini_errors", "Errors encountered", ["type"])
        return cls._instance

    def track_request(self, latency: float, tokens: int):
        self.request_latency.observe(latency)
        self.tokens_used.inc(tokens)

    def track_error(self, error_type: str):
        self.errors.labels(type=error_type).inc()

class PromptTemplate:
    def __init__(self, template: str, variables: List[str]):
        self.template = template
        self.variables = variables

    def format(self, **kwargs) -> str:
        return self.template.format(**kwargs)

class Chain:
    def __init__(self, steps: List[Callable]):
        self.steps = steps

    async def execute(self, initial_input: Any) -> Any:
        result = initial_input
        for step in self.steps:
            result = await step(result)
        return result

class ModelManager:
    MODELS = {
        "gemini-2.0-flash": {"cost": 0.001, "speed": "fast", "quality": "high", "multimodal": True},
        "gemini-2.0-flash-lite-preview-02-05": {"cost": 0.0005, "speed": "very_fast", "quality": "medium", "multimodal": True},
        "gemini-2.0-pro-exp-02-05": {"cost": 0.002, "speed": "slow", "quality": "very_high", "multimodal": True},
        "gemini-2.0-flash-thinking-exp-01-21": {"cost": 0.0015, "speed": "medium", "quality": "high", "multimodal": True},
    }

    def select_model(self, requirements: Dict[str, Any]) -> str:
        if not requirements:
            return "gemini-2.0-flash"
        cost_weight = 2.0  # Give more importance to cost
        best_model = None
        min_score = float("inf")
        for model, meta in self.MODELS.items():
            score = self._calculate_score(model, meta, requirements)
            if score < min_score:
                min_score = score
                best_model = model
        return best_model

    def _calculate_score(self, model, meta, requirements):
        cost_score = abs(meta["cost"] - requirements.get("max_cost", 0))
        speed_score = self._map_speed(meta["speed"], requirements.get("min_speed", "slow"))
        quality_score = self._map_quality(meta["quality"], requirements.get("min_quality", "medium"))
        return cost_score + speed_score + quality_score

    def _map_speed(self, current, target):
        speeds = ["very_slow", "slow", "medium", "fast", "very_fast"]
        return abs(speeds.index(current) - speeds.index(target))

    def _map_quality(self, current, target):
        qualities = ["low", "medium", "high", "very_high"]
        return abs(qualities.index(current) - qualities.index(target))

class RetryHandler:
    def __init__(self, max_retries: int = 3, backoff_factor: float = 1.5):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

    async def execute_with_retry(self, func, *args, **kwargs):
        for attempt in range(self.max_retries):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Error encountered, retrying (Attempt {attempt + 1}): {e}")
                await asyncio.sleep(self.backoff_factor ** attempt)
        raise Exception("Max retries exceeded")

#################################
# GeminiClient Implementation
#################################

class GeminiClient:
    def __init__(
        self,
        key_rotation_interval: float = 0.5,
        history_size: int = 4,
        model: str = "gemini-2.0-flash",
        generation_config: Optional[Dict[str, Any]] = None,
        improve_prompts: bool = False,
        cache_dir: str = "./cache",
    ):
        self._load_api_keys()
        self.key_rotation_interval = key_rotation_interval
        self.conversation_history = deque(maxlen=history_size)
        self.model = model
        self.generation_config = generation_config or {
            "safety_settings": [s.__dict__ for s in self._default_safety_settings()],
            "max_output_tokens": 1024,
            "temperature": 0.7,
        }
        self.improve_prompts = improve_prompts
        self.request_counter = 0
        self.cache = Cache(cache_dir)
        self.metrics = MetricsCollector()
        self.key_manager = KeyManager(self.api_keys, capacity=100, refill_rate=50, secret=self._get_key_secret())
        self.retry_handler = RetryHandler()
        self.model_manager = ModelManager()
        self.last_key_rotation = time.time()
        self._initialize_client()

    def _get_key_secret(self) -> str:
        """Retrieve the secret key for encryption/decryption from environment."""
        return os.getenv("KEY_SECRET", "default_secret")

    def _load_api_keys(self):
        """Load and encrypt API keys from .env file that match GEMINI_API_KEY pattern with numbers."""
        load_dotenv()
        self.api_keys = []
        gemini_keys = {k: v for k, v in os.environ.items() if k.startswith('GEMINI_API_KEY')}
        sorted_keys = sorted(gemini_keys.items(), key=lambda x: int(''.join(filter(str.isdigit, x[0]))))
        secret = self._get_key_secret()
        for _, key in sorted_keys:
            if key.startswith('"'):
                key = key.strip('"')
            if key.startswith('AIzaSy'):
                encrypted_key = encrypt_key(key, secret)
                self.api_keys.append(encrypted_key)
        if not self.api_keys:
            raise ValueError("No valid Gemini API keys found in .env file")
        logger.info(f"Loaded {len(self.api_keys)} API keys")

    def _default_safety_settings(self):
        return [
            types.SafetySetting(
                category=category,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            )
            for category in [
                types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
            ]
        ]

    def _initialize_client(self):
        # Force rotation to the next key at startup.
        self.current_key = self.key_manager.get_next_key()
        self.client = genai.Client(api_key=self.current_key)
        self.last_key_rotation = time.time()
        logger.info(f"Initialized client with API key {self.current_key[:8]}...")

    def enable_code_execution(self):
        self.generation_config.setdefault("tools", []).append(types.ToolCodeExecution())

    def enable_google_search(self):
        self.generation_config.setdefault("tools", []).append(types.GoogleSearch())

    def enable_retrieval(self):
        self.generation_config.setdefault("tools", []).append(types.Retrieval())

    def enable_function_calling(self, functions: List[types.FunctionDeclaration]):
        self.generation_config["function_declarations"] = functions

    async def improve_prompt(self, prompt: str) -> str:
        if not self.improve_prompts:
            return prompt
        meta_prompt = self._build_meta_prompt()
        # Remove extra field before constructing config
        extra_fields = {}
        if "function_declarations" in self.generation_config:
            extra_fields["function_declarations"] = self.generation_config.pop("function_declarations")
        config_dict = types.GenerateContentConfig(**self.generation_config).model_dump()
        if extra_fields:
            config_dict.update(extra_fields)
        response = await asyncio.to_thread(
            self.client.models.generate_content,
            model=self.model,
            contents=[f"{meta_prompt}\nTask, objective, or current prompt:\n{prompt}"],
            config=config_dict,
        )
        return response.text.strip() if hasattr(response, 'text') else str(response)

    async def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """Process multiple prompts in parallel while respecting rate limits."""
        results = []
        for prompt in prompts:
            try:
                result = await self.generate_content_async(prompt, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Batch processing error for prompt: {prompt[:50]}... Error: {str(e)}")
                results.append(None)
        # Filter out None responses
        return [r if r is not None else "" for r in results]

    async def generate_content_async(self, content: Union[str, List[Union[str, PIL.Image.Image]]], **kwargs) -> str:
        history = self._build_history_prompt()
        if isinstance(content, list):
            full_prompt = [history] + content if history else content
        else:
            full_prompt = [f"{history}\n{content}"] if history else [content]

        cache_key = self._compute_cache_key(full_prompt, kwargs)
        cached_response = self.cache.get(cache_key)
        if cached_response:
            self._update_history(str(content), cached_response)
            self.metrics.track_request(0, 0)
            return cached_response

        # Force rotation to a new key at the start of the request.
        self.current_key = self.key_manager.get_next_key()
        self.client = genai.Client(api_key=self.current_key)
        self.last_key_rotation = time.time()
        tried_keys = {self.current_key}
        last_error = None

        while True:
            try:
                extra_fields = {}
                if "function_declarations" in self.generation_config:
                    extra_fields["function_declarations"] = self.generation_config.pop("function_declarations")
                config_dict = types.GenerateContentConfig(**self.generation_config).model_dump()
                if extra_fields:
                    config_dict.update(extra_fields)
                response = await asyncio.to_thread(
                    self.client.models.generate_content,
                    model=self.model,
                    contents=full_prompt,
                    config=config_dict,
                )
                response_text = response.text if hasattr(response, 'text') else str(response)
                self._update_history(str(content), response_text)
                self.cache.set(cache_key, response_text)
                self.metrics.track_request(getattr(response, 'latency', 0), len(str(full_prompt)))
                return response_text
            except Exception as e:
                last_error = e
                if "API key not valid" in str(e):
                    logger.warning(f"Invalid API key {self.current_key[:8]}...")
                elif "model not found" in str(e).lower():
                    logger.warning(f"Model {self.model} not found, falling back to gemini-pro")
                    self.model = "gemini-pro"
                    continue
                else:
                    logger.warning(f"Request failed: {str(e)}")
                if len(tried_keys) >= len(self.api_keys):
                    break
                await asyncio.sleep(self.key_rotation_interval)
                new_key = self.key_manager.get_next_key()
                if new_key in tried_keys:
                    continue
                self.current_key = new_key
                self.client = genai.Client(api_key=self.current_key)
                tried_keys.add(new_key)
                self.last_key_rotation = time.time()
                logger.info(f"Rotated to API key: {self.current_key[:8]}...")
        raise Exception(f"All API keys ({len(self.api_keys)}) exhausted. Last error: {str(last_error)}")

    def generate_content_stream(self, content: Union[str, List[Union[str, PIL.Image.Image]]], **kwargs) -> Generator:
        async def stream_generator():
            try:
                history = self._build_history_prompt()
                full_prompt = [f"{history}\n{content}"] if history else [content]
                extra_fields = {}
                if "function_declarations" in self.generation_config:
                    extra_fields["function_declarations"] = self.generation_config.pop("function_declarations")
                config_dict = types.GenerateContentConfig(**self.generation_config).model_dump()
                if extra_fields:
                    config_dict.update(extra_fields)
                response = await asyncio.to_thread(
                    self.client.models.generate_content,
                    model=self.model,
                    contents=full_prompt,
                    config=config_dict,
                )
                if hasattr(response, '__aiter__'):
                    async for chunk in response:
                        if hasattr(chunk, 'text') and chunk.text:
                            yield chunk.text
                            await asyncio.sleep(0.01)
                elif hasattr(response, '__iter__'):
                    for chunk in response:
                        if hasattr(chunk, 'text') and chunk.text:
                            yield chunk.text
                            await asyncio.sleep(0.01)
                else:
                    yield str(response)
            except Exception as e:
                logger.error(f"Streaming error: {str(e)}")
                yield ""
        return stream_generator()

    def _compute_cache_key(self, prompt: Union[str, List], config: dict) -> str:
        combined = f"{prompt}{json.dumps(config, sort_keys=True)}"
        return hashlib.sha256(combined.encode()).hexdigest()

    def _build_history_prompt(self) -> str:
        """
        Constructs a conversation history prompt by concatenating past user inputs and assistant responses.
        Ensures that None responses are replaced with an empty string.
        """
        if not self.conversation_history:
            return ""
        history_parts = []
        for user, resp in self.conversation_history:
            history_parts.extend([
                "User: " + user,
                "Assistant: " + (resp if resp is not None else "")
            ])
        return "\n".join(history_parts)

    def _update_history(self, user_input: str, response: str):
        """
        Appends the user input and response to the conversation history.
        Ensures that a None response is converted to an empty string.
        """
        self.conversation_history.append((user_input, response if response is not None else ""))

    def set_generation_config(self, config: Dict[str, Any]):
        self.generation_config = config

    def toggle_improve_prompts(self, enable: bool):
        self.improve_prompts = enable

    def _build_meta_prompt(self) -> str:
        return """
        Improve the given prompt to make it more effective for the language model.
        - Understand the task: Clarify the main objective, goals, requirements, constraints, and expected output.
        - Reasoning before conclusions: Encourage reasoning steps before reaching conclusions. If the user's example has conclusions first, reverse the order.
        - Examples: Include high-quality examples if helpful, using [placeholders] for complex elements.
        - Clarity and conciseness: Use clear and specific language. Avoid unnecessary instructions or vague statements.
        - Formatting: Use Markdown features to improve readability. Avoid code blocks unless explicitly required.
        """

#################################
# Additional Classes
#################################

class ResponseValidator:
    def __init__(self, schema: Dict):
        self.schema = schema

    def validate_and_transform(self, output: str) -> Any:
        try:
            data = json.loads(output)
            return data
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON response")

class WebhookHandler:
    async def handle_webhook(self, payload: Dict) -> None:
        client_instance = GeminiClient()
        response = await client_instance.generate_content_async(payload["prompt"])
        # Send back response as per webhook requirements

class PromptRequest(BaseModel):
    prompt: str

class APIWrapper:
    def __init__(self, gemini_client: GeminiClient, host: str = "127.0.0.1", port: int = 8000):
        self.app = FastAPI()
        self.gemini_client = gemini_client
        self.setup_routes()

    def setup_routes(self):
        @self.app.post("/generate")
        async def generate(request: PromptRequest):
            return {"response": await self.gemini_client.generate_content_async(request.prompt)}

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            try:
                while True:
                    data = await websocket.receive_text()
                    response = await self.gemini_client.generate_content_async(data)
                    await websocket.send_text(response)
            except WebSocketDisconnect:
                logger.info("WebSocket disconnected by client.")
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
            finally:
                try:
                    await websocket.close()
                except Exception as e:
                    logger.error(f"Error while closing WebSocket: {e}")

    async def start_websocket_handler(self):
        uvicorn.run(self.app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    client_instance = GeminiClient(
        key_rotation_interval=3,
        improve_prompts=True,
        cache_dir="./gemini_cache"
    )

    async def example_async():
        response = await client_instance.generate_content_async("Explain quantum computing")
        print(response)

    def example_stream():
        for chunk in client_instance.generate_content_stream("Tell a story"):
            print(chunk, end="")

    api_wrapper = APIWrapper(client_instance)
    uvicorn.run(api_wrapper.app, host="0.0.0.0", port=8000)
