"""
LLM Provider Factory
"""

import os
from src.env import load_env
from src.rag.llm.base import BaseLLMProvider
from src.rag.llm.gemini_provider import GeminiProvider
from src.rag.llm.openai_provider import OpenAIProvider
from src.rag.llm.ollama_provider import OllamaProvider
from src.rag.llm.mock_provider import MockProvider


def resolve_model_name(provider: str) -> str | None:
    """
    <PROVIDER>_MODEL wins; LLM_MODEL applies only to the provider named by
    LLM_PROVIDER, so a Gemini model id is never sent to OpenAI/Ollama.
    None lets the provider use its own default.
    """
    specific = os.getenv(f"{provider.upper()}_MODEL", "").strip()
    if specific:
        return specific
    if os.getenv("LLM_PROVIDER", "").strip().lower() == provider:
        return os.getenv("LLM_MODEL", "").strip() or None
    return None


def get_llm_provider(provider_name: str = None) -> BaseLLMProvider:
    load_env()
    name =(provider_name or os.getenv("LLM_PROVIDER", "gemini")).lower()

    if name == "gemini":
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            print("! Warning: GEMINI_API_KEY chưa được cấu hình. Chuyển sang MockProvider để test.")
            return MockProvider()
        return GeminiProvider(api_key=api_key, model_name=resolve_model_name(name))
    elif name == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            print("! Warning: OPENAI_API_KEY chưa được cấu hình. Chuyển sang MockProvider để test.")
            return MockProvider()
        return OpenAIProvider(api_key=api_key, model_name=resolve_model_name(name))
    elif name == "ollama":
        return OllamaProvider(model_name=resolve_model_name(name))
    elif name == "mock":
        return MockProvider()
    else:
        print(f"! Provider '{name}' không hợp lệ. Sử dụng MockProvider mặc định.")
        return MockProvider()
