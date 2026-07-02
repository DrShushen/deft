import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional

import openai
from aiohttp import ClientSession
from omegaconf import DictConfig

from src.llm.engines.generic_engine import GenericEngine

logger = logging.getLogger(__name__)


class RequestAdapter(ABC):
    """Adapter that maps generic generation parameters to provider-specific kwargs."""

    @abstractmethod
    def build_request_kwargs(
        self,
        *,
        model: str,
        messages: list,
        temperature: float,
        top_p: float,
        max_tokens: int,
        n_generations_per_prompt: int,
        request_timeout: int,
    ) -> Dict:
        """Build request kwargs."""
        pass


class StandardRequestAdapter(RequestAdapter):
    """Default chat-completions request shape."""

    def build_request_kwargs(
        self,
        *,
        model: str,
        messages: list,
        temperature: float,
        top_p: float,
        max_tokens: int,
        n_generations_per_prompt: int,
        request_timeout: int,
    ) -> Dict:
        """Build request kwargs."""
        return {
            "engine": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "n": n_generations_per_prompt,
            "request_timeout": request_timeout,
            "stop": " ##",
        }


class GPT5RequestAdapter(RequestAdapter):


    def __init__(self, n: int = 8, reasoning_effort: str = "minimal") -> None:
        """Initialize the instance."""
        self.n = n
        self.reasoning_effort = reasoning_effort

    def build_request_kwargs(
        self,
        *,
        model: str,
        messages: list,
        temperature: float,
        top_p: float,
        max_tokens: int,
        n_generations_per_prompt: int,
        request_timeout: int,
    ) -> Dict:
        """Build request kwargs."""
        return {
            "engine": model,
            "messages": messages,
            "max_completion_tokens": max_tokens,
            "n": self.n,
            "request_timeout": request_timeout,
            "reasoning_effort": self.reasoning_effort,
        }


class GptOssRequestAdapter(RequestAdapter):
   

    def __init__(self, reasoning_effort: str = "medium") -> None:
        """Initialize the instance."""
        self.reasoning_effort = reasoning_effort

    def build_request_kwargs(
        self,
        *,
        model: str,
        messages: list,
        temperature: float,
        top_p: float,
        max_tokens: int,
        n_generations_per_prompt: int,
        request_timeout: int,
    ) -> Dict:
        """Build request kwargs."""
        return {
            "engine": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "n": n_generations_per_prompt,
            "request_timeout": request_timeout,
            "reasoning_effort": self.reasoning_effort,
            "stop": " ##",
        }


class UnifiedAPIEngine(GenericEngine):
    """Single OpenAI-compatible engine with pluggable request adapters."""

    def __init__(
        self,
        content: str,
        temperature: float = 0.75,
        top_p: float = 0.95,
        max_tokens: int = 8000,
        max_attempts: int = 100,
        rate_limiter=None,
        request_timeout: int = 20,
        config: DictConfig = None,
        request_adapter: RequestAdapter = None,
    ) -> None:
        """Initialize the instance."""
        super().__init__(
            content=content,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            max_attempts=max_attempts,
            rate_limiter=rate_limiter,
            config=config,
        )
        self.request_timeout = request_timeout
        self.request_adapter = request_adapter or StandardRequestAdapter()

    @staticmethod
    def _cfg_get(cfg, key: str):
        """Handle cfg get."""
        if isinstance(cfg, dict):
            return cfg[key]
        return getattr(cfg, key)

    @staticmethod
    def _is_openai_api_type(api_type: Optional[str]) -> bool:
        """Check whether openai api type."""
        if api_type is None:
            return False
        return str(api_type).lower() in {"open_ai", "openai"}

    @classmethod
    def _normalize_request_kwargs_for_provider(
        cls, request_kwargs: Dict, api_type: Optional[str]
    ) -> Dict:
        """
        Keep adapters provider-agnostic by normalizing final kwargs before dispatch:
        - OpenAI API prefers `model`, not `engine`.
        - Azure OpenAI requires `engine` (deployment name).
        """
        normalized = dict(request_kwargs)
        if cls._is_openai_api_type(api_type):
            if "engine" in normalized and "model" not in normalized:
                normalized["model"] = normalized.pop("engine")
        return normalized

    def _resolve_llm_models_cfg(self):
        """
        Support both config shapes:
        1) nested: config.llm_models.{model, api_type, ...}
        2) flat:   config.{model, api_type, ...}
        """
        if self.config is None:
            raise ValueError("Missing engine config.")

        if isinstance(self.config, dict):
            return self.config.get("llm_models", self.config)

        try:
            nested = self.config.get("llm_models")
            if nested is not None:
                return nested
        except Exception:
            pass

        try:
            nested = self.config.llm_models
            if nested is not None:
                return nested
        except Exception:
            pass

        return self.config

    async def _async_generate(self, user_message: str, n_generations_per_prompt: int):
        """Generate a response from the LLM asynchronously."""
        llm_models_cfg = self._resolve_llm_models_cfg()
        api_type = self._cfg_get(llm_models_cfg, "api_type")
        openai.api_type = api_type
        openai.api_base = self._cfg_get(llm_models_cfg, "api_base")
        openai.api_version = self._cfg_get(llm_models_cfg, "api_version")
        openai.api_key = self._cfg_get(llm_models_cfg, "api_key")

        messages = [
            {"role": "system", "content": self.content},
            {"role": "user", "content": user_message},
        ]

        session = ClientSession(trust_env=True)
        if hasattr(openai, "aiosession") and hasattr(openai.aiosession, "set"):
            openai.aiosession.set(session)

        resp = None
        try:
            for attempt in range(self.MAX_RETRIES):
                try:
                    request_kwargs = self.request_adapter.build_request_kwargs(
                        model=self._cfg_get(llm_models_cfg, "model"),
                        messages=messages,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        max_tokens=self.max_tokens,
                        n_generations_per_prompt=n_generations_per_prompt,
                        request_timeout=self.request_timeout,
                    )
                    request_kwargs = self._normalize_request_kwargs_for_provider(
                        request_kwargs=request_kwargs,
                        api_type=api_type,
                    )
                    resp = await openai.ChatCompletion.acreate(**request_kwargs)
                    break
                except (
                    openai.error.AuthenticationError,
                    openai.error.PermissionError,
                    openai.error.InvalidRequestError,
                ) as e:
                    # Non-retryable: bad key / no access / wrong deployment or model
                    # (e.g. Azure DeploymentNotFound). Retrying cannot help, and
                    # silently returning None here lets DEFT fall back to naive
                    # features and "succeed" on a dead / misconfigured LLM. Fail loudly.
                    logger.error("[LLM API] Non-retryable API error, aborting: %s", e)
                    raise
                except openai.error.RateLimitError as e:
                    if attempt < self.MAX_RETRIES - 1:
                        retry_time = self._extract_retry_time(str(e), attempt)
                        logger.info(
                            f"[LLM API] Rate Limit Error. Retrying in {retry_time} seconds"
                        )
                        await asyncio.sleep(retry_time)
                except asyncio.exceptions.TimeoutError as e:
                    if attempt < self.MAX_RETRIES - 1:
                        logger.info(e)
                        logger.info(
                            f"[LLM API] OpenAI API timeout. Sleeping for {self.retry_backoff[attempt]} seconds"
                        )
                        await asyncio.sleep(self.retry_backoff[attempt])
                except openai.error.Timeout as e:
                    if attempt < self.MAX_RETRIES - 1:
                        logger.info(e)
                        logger.info(
                            f"[LLM API] OpenAI API timeout. Sleeping for {self.retry_backoff[attempt]} seconds"
                        )
                        await asyncio.sleep(self.retry_backoff[attempt])
                except Exception as e:
                    if attempt == self.MAX_RETRIES - 1:
                        logger.info(f"[LLM API] Failed to generate response, exiting: {e}")
                        return None
        finally:
            await session.close()

        return resp

    async def query_llm(self, list_prompts: list, n_generations_per_prompt: int):
        """Perform concurrent generation of responses from the LLM async."""
        coroutines = [
            self._async_generate(prompt, n_generations_per_prompt)
            for prompt in list_prompts
        ]
        tasks = [asyncio.create_task(c) for c in coroutines]
        llm_responses = await asyncio.gather(*tasks)

        results = [None] * len(coroutines)
        for idx, response in enumerate(llm_responses):
            if response is not None:
                results[idx] = response
        return results
