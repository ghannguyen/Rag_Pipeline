"""
Gemini Provider using official google-genai SDK.
"""

import os
from typing import Any, Dict, List

from src.rag.llm.base import BaseLLMProvider


class GeminiProvider(BaseLLMProvider):
    DEFAULT_MODEL = "gemini-3.6-flash"

    def __init__(
        self,
        api_key: str = None,
        model_name: str = None,
    ):
        """
        Initialize Gemini provider.

        Priority:
        1. model_name passed directly (the factory resolves env vars)
        2. Default: gemini-3.6-flash
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.model_name = model_name or self.DEFAULT_MODEL

    def generate(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> str:
        """
        Generate an answer using the official Google GenAI SDK.
        """

        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY không được để trống. "
                "Vui lòng export GEMINI_API_KEY trước khi chạy."
            )

        system_instruction = ""
        user_parts = []

        # Convert message format used by the RAG pipeline
        # into Gemini system instruction + user prompt.
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if not content:
                continue

            if role == "system":
                system_instruction = content

            elif role == "user":
                user_parts.append(content)

            elif role == "assistant":
                # Preserve assistant content if the pipeline ever
                # passes multi-turn messages.
                user_parts.append(f"Assistant: {content}")

        user_prompt = "\n\n".join(user_parts).strip()

        if not user_prompt:
            raise ValueError(
                "Không có user prompt để gửi tới Gemini."
            )

        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=self.api_key)

            config = types.GenerateContentConfig(
                system_instruction=(
                    system_instruction
                    if system_instruction
                    else None
                ),
                temperature=kwargs.get("temperature", 0.1),
                max_output_tokens=kwargs.get("max_tokens", 1200),
            )

            print(
                f"Calling Gemini API using model: {self.model_name}"
            )

            response = client.models.generate_content(
                model=self.model_name,
                contents=user_prompt,
                config=config,
            )

            if response is None:
                raise RuntimeError(
                    "Gemini API không trả về response."
                )

            try:
                answer = response.text
            except Exception as exc:
                raise RuntimeError(
                    "Gemini có trả response nhưng không có text output."
                ) from exc

            if not answer or not answer.strip():
                raise RuntimeError(
                    "Gemini API trả về nội dung rỗng."
                )

            print(
                f"✓ Gemini API response received using model: "
                f"{self.model_name}"
            )

            return answer.strip()

        except ImportError as exc:
            raise RuntimeError(
                "Không import được google-genai SDK. "
                "Kiểm tra package google-genai trong .venv."
            ) from exc

        except Exception as exc:
            err_msg = str(exc)

            if "401" in err_msg or "UNAUTHENTICATED" in err_msg:
                raise RuntimeError(
                    "GEMINI_API_KEY không hợp lệ hoặc không được "
                    "Google xác thực."
                ) from exc

            if "403" in err_msg or "PERMISSION_DENIED" in err_msg:
                raise RuntimeError(
                    "Gemini API từ chối quyền truy cập của project/API key "
                    f"cho model {self.model_name}. Chi tiết: {err_msg}"
                ) from exc

            if "404" in err_msg or "NOT_FOUND" in err_msg:
                raise RuntimeError(
                    f"Gemini model '{self.model_name}' không khả dụng "
                    f"cho API key/project hiện tại. Chi tiết: {err_msg}"
                ) from exc

            if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                raise RuntimeError(
                    "Gemini API đã vượt quota/rate limit. "
                    f"Chi tiết: {err_msg}"
                ) from exc

            raise RuntimeError(
                f"Lỗi gọi Gemini SDK: {err_msg}"
            ) from exc