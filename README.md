# Better GenAI 🚀  

Welcome to **Better GenAI**, a modern and robust Python client library for interacting with the Google Gemini API! Created by **Chun**, this project is designed to streamline AI-powered content generation, with features like secure API key management, conversation memory, caching, streaming, and FastAPI integration. 😎  

---

## 📌 Table of Contents  

- [Overview](#overview)  
- [Features](#features)  
- [Project Structure](#project-structure)  
- [Installation](#installation)  
- [Configuration](#configuration)  
- [Client Library Usage](#client-library-usage)  
  - [Initializing the Client](#initializing-the-client)  
  - [Basic Text Generation](#basic-text-generation)  
  - [Conversation Memory](#conversation-memory)  
  - [Image-based Prompting](#image-based-prompting)  
  - [Streaming Responses](#streaming-responses)  
  - [Batch Text Generation](#batch-text-generation)  
  - [Prompt Improvement](#prompt-improvement)  
  - [Enabling Additional Tools](#enabling-additional-tools)  
  - [Model Selection](#model-selection)  
- [FastAPI Server Usage](#fastapi-server-usage)  
- [Running Tests](#running-tests)  
- [Contributing](#contributing)  
- [License](#license)  

---

## 🌟 Overview  

**Better GenAI** is a Python-based client library built on top of the new `google-genai` module (the successor to `google-generativeai`). It provides an easy-to-use and powerful interface for AI-driven content generation, handling:  

✅ **Secure API Key Management** – Automatic key encryption and rotation to avoid rate limits.  
✅ **Conversation Memory** – Maintains a short history of interactions for contextual responses.  
✅ **Caching & Metrics** – Optimizes performance by caching responses and tracking latency & errors.  
✅ **Multi-modal Support** – Works with both text and image inputs.  
✅ **Streaming Responses** – Supports real-time content generation.  
✅ **FastAPI Integration** – Offers HTTP and WebSocket endpoints for AI-driven applications.  

---

## 📁 Project Structure  

After cloning the repository, your project directory should look like this:  

```
project
│
├── better_genai
│   ├── .env                      # API keys and encryption secret
│   ├── client.py             # Core GeminiClient implementation
│   ├── requirements.txt      # Dependencies
│   ├── sample_usage.py       # Example usage of the GeminiClient
│   ├── test.png              # Sample image for image-based prompts
│   ├── test_client.py        # Unit tests
└── main.py                   # Runs FastAPI server
```

---

## 🔧 Installation  

1️⃣ **Clone the Repository:**  

```bash
git clone https://github.com/chungus1310/better_genai.git
cd better_genai
```

2️⃣ **Set Up a Virtual Environment (Optional but Recommended):**  

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3️⃣ **Install Dependencies:**  

```bash
pip install -r requirements.txt
```

---

## 🔑 Configuration  

Before running the project, create a `.env` file in the **project** directory with the following variables:  

```
# .env file
GEMINI_API_KEY1="AIzaSy_your_first_api_key"
GEMINI_API_KEY2="AIzaSy_your_second_api_key"
# Add more keys as needed...

KEY_SECRET="your_secret_key"  # Used to encrypt API keys
```

Make sure your API keys start with `AIzaSy`, as expected by the client.  

---

# 📚 Client Library Usage  

The **GeminiClient** class in `client.py` provides the main interface for interacting with the Gemini API.  

## 🔹 Initializing the Client  

```python
from better_genai.client import GeminiClient

client = GeminiClient(
    key_rotation_interval=1.0,  # Rotate API keys every 1 second
    history_size=4,             # Keep last 4 messages in conversation memory
    model="gemini-2.0-flash",   # Default model
    improve_prompts=True,       # Enable prompt improvement
    cache_dir="./cache"         # Enable caching
)
```

## 🔹 Basic Text Generation  

```python
response = client.generate_content_async("What is quantum computing?")
print(response)
```

## 🔹 Conversation Memory  

The client automatically remembers past interactions (up to `history_size`).  

```python
client.generate_content_async("Hello!")
response = client.generate_content_async("What did I just say?")
print(response)  # Should refer to "Hello!"
```

## 🔹 Image-based Prompting  

Supports **multi-modal inputs** (text + images).  

```python
from PIL import Image

image = Image.open("test.png")
response = client.generate_content_async(["Describe this image:", image])
print(response)
```

## 🔹 Streaming Responses  

For **real-time** response generation:  

```python
for chunk in client.generate_content_stream("Tell me a sci-fi story"):
    print(chunk, end="", flush=True)
```

## 🔹 Batch Text Generation  

Process **multiple prompts in parallel** for efficiency.  

```python
prompts = [
    "Explain relativity",
    "Summarize AI history",
    "Future of robotics?"
]

responses = client.generate_batch(prompts)
print(responses)
```

## 🔹 Prompt Improvement  

Enhance prompt clarity **before** sending it to the model.  

```python
optimized_prompt = client.improve_prompt("tell me what deep learning is")
print(optimized_prompt)
```

## 🔹 Enabling Additional Tools  

You can enable **special capabilities** like:  

```python
client.enable_code_execution()
client.enable_google_search()
client.enable_retrieval()
```

## 🔹 Model Selection  

The **ModelManager** selects the best model based on cost, speed, and quality.  

```python
best_model = client.model_manager.select_model({
    "max_cost": 0.001, 
    "min_speed": "fast", 
    "min_quality": "high"
})
print(best_model)  # Outputs the best available model
```

---

# 🚀 FastAPI Server Usage  

The project also includes a **FastAPI wrapper**, allowing the client to be used as a **web service**.  

## 🔹 Start the Server  

Run the `main.py` file:  

```bash
python main.py
```

## 🔹 HTTP API Usage  

Send a **POST** request with a prompt:  

```bash
curl -X POST "http://127.0.0.1:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Tell me a fun fact about space."}'
```

## 🔹 WebSocket API Usage  

Connect via WebSocket:  

```javascript
let ws = new WebSocket("ws://127.0.0.1:8000/ws");
ws.onopen = () => ws.send("Tell me a joke!");
ws.onmessage = (event) => console.log("Received:", event.data);
```

---

# ✅ Running Tests  

To validate everything is working correctly, run:  

```bash
python better_genai/test_client.py
```

This tests:  

- **Basic text & image generation**  
- **Conversation memory**  
- **Streaming responses**  
- **API key rotation**  
- **FastAPI HTTP & WebSocket endpoints**  

---

# 🤝 Contributing  

We ❤️ contributions!  

1. Fork the repository.  
2. Create a feature branch (`git checkout -b feature-name`).  
3. Commit your changes (`git commit -m "Add new feature"`).  
4. Push to the branch (`git push origin feature-name`).  
5. Open a pull request.  

---

# 📜 License  

This project is open-source. See the `LICENSE` file for details.  

---

🚀 **Enjoy using Better GenAI!** Happy coding! 😃🎉  

---  

*Created with ❤️ by [Chun](https://github.com/chungus1310).*