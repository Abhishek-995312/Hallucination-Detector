"""One function that talks to the LLM. Swap the provider here and nothing else changes.

Setup (Groq, free tier):
    1. Create an API key at https://console.groq.com
    2. Create a file named .env in the project root (next to src/) containing:
         GROQ_API_KEY=your_key_here
    3. Never commit .env to GitHub (add it to .gitignore).
"""

import os
import time

from dotenv import load_dotenv
from groq import Groq

from src import config

# Read GROQ_API_KEY (and optional LLM_MODEL) from <project root>/.env
load_dotenv(config.ROOT / ".env")

# Check https://console.groq.com/docs/models for currently available models.
MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")

_client = None


def call_llm(prompt: str, system: str | None = None,
             temperature: float = 0.0, max_tokens: int = 600) -> str:
    global _client
    if _client is None:
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise RuntimeError("GROQ_API_KEY not found. Add it to the .env file in the project root.")
        _client = Groq(api_key=key)

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    for attempt in range(3):
        try:
            resp = _client.chat.completions.create(
                model=MODEL, messages=messages,
                temperature=temperature, max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:  # rate limit or temporary error: wait and retry
            if attempt == 2:
                raise
            print(f"  LLM error ({e}); retrying...")
            time.sleep(5 * (attempt + 1))

