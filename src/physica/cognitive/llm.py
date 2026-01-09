"""LLM backend abstraction for cognitive layer.

Provides unified interface to different LLM providers with CrewAI compatibility.
Supports: OpenAI, Claude, Watsonx, Ollama, and Mock (no API required).
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .settings import get_settings, LLMProvider


@dataclass
class LLMMessage:
    """A message in LLM conversation."""

    role: str  # 'system', 'user', or 'assistant'
    content: str


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""

    @abstractmethod
    def generate(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> str:
        """Generate a response from the LLM.

        Parameters
        ----------
        messages:
            Conversation history.
        temperature:
            Sampling temperature.
        max_tokens:
            Maximum response length.

        Returns
        -------
        response:
            LLM generated text.
        """
        pass


class MockLLM(LLMBackend):
    """Mock LLM for testing and demos (no API calls).

    Returns predefined responses for common physics queries.
    """

    def generate(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> str:
        """Generate mock response based on last message."""
        if not messages:
            return "No input provided."

        last_message = messages[-1].content.lower()

        # Pattern matching for common physics queries
        if "projectile" in last_message or "trajectory" in last_message:
            return """
{
  "intent": "simulate_projectile",
  "parameters": {
    "initial_velocity": 50.0,
    "angle_degrees": 45.0,
    "mass": 1.0,
    "drag_coefficient": 0.05
  },
  "objectives": ["maximize_range", "hit_target"],
  "constraints": ["energy_conservation", "realistic_parameters"]
}
"""
        elif "optimize" in last_message:
            return """
{
  "intent": "optimize_trajectory",
  "parameters": {
    "target_distance": 300.0,
    "angle_degrees": 45.0,
    "tolerance": 2.0
  },
  "objectives": ["minimize_error"],
  "constraints": ["energy_conservation", "momentum_conservation"]
}
"""
        elif "temperature" in last_message or "heat" in last_message:
            return """
{
  "intent": "simulate_heat_transfer",
  "parameters": {
    "thermal_diffusivity": 1.0,
    "initial_temp": 100.0,
    "boundary_temp": 0.0,
    "domain_size": [1.0, 1.0]
  },
  "objectives": ["steady_state"],
  "constraints": ["energy_conservation", "heat_equation"]
}
"""
        else:
            return """
{
  "intent": "analyze_system",
  "parameters": {},
  "objectives": ["understand_physics"],
  "constraints": ["fundamental_laws"]
}
"""


class ClaudeLLM(LLMBackend):
    """Anthropic Claude backend."""

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-5-sonnet-20241022"):
        """Initialize Claude backend.

        Parameters
        ----------
        api_key:
            Anthropic API key. If None, reads from ANTHROPIC_API_KEY env var.
        model:
            Claude model to use.
        """
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic package required for Claude backend. "
                "Install with: pip install 'physica[llm]'"
            )

        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = model

    def generate(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> str:
        """Generate response using Claude."""
        # Convert to Anthropic message format
        formatted_messages = []
        system_message = None

        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            else:
                formatted_messages.append({
                    "role": msg.role,
                    "content": msg.content,
                })

        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": formatted_messages,
        }

        if system_message:
            kwargs["system"] = system_message

        response = self.client.messages.create(**kwargs)
        return response.content[0].text


class OpenAILLM(LLMBackend):
    """OpenAI GPT backend."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        """Initialize OpenAI backend.

        Parameters
        ----------
        api_key:
            OpenAI API key. If None, reads from OPENAI_API_KEY env var.
        model:
            OpenAI model to use.
        """
        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai package required for OpenAI backend. "
                "Install with: pip install 'physica[llm]'"
            )

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.client = openai.OpenAI(api_key=self.api_key)
        self.model = model

    def generate(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> str:
        """Generate response using OpenAI."""
        formatted_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=formatted_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return response.choices[0].message.content


class WatsonxLLM(LLMBackend):
    """IBM Watsonx.ai backend."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        project_id: Optional[str] = None,
        model: str = "meta-llama/llama-3-3-70b-instruct",
        base_url: str = "https://us-south.ml.cloud.ibm.com",
    ):
        """Initialize Watsonx backend.

        Parameters
        ----------
        api_key:
            IBM Cloud API key. If None, reads from WATSONX_API_KEY env var.
        project_id:
            Watsonx project ID. If None, reads from WATSONX_PROJECT_ID env var.
        model:
            Model ID to use.
        base_url:
            Watsonx API base URL.
        """
        try:
            from ibm_watsonx_ai import Credentials
            from ibm_watsonx_ai.foundation_models import ModelInference
        except ImportError:
            raise ImportError(
                "ibm-watsonx-ai package required for Watsonx backend. "
                "Install with: pip install ibm-watsonx-ai"
            )

        self.api_key = api_key or os.getenv("WATSONX_API_KEY")
        self.project_id = project_id or os.getenv("WATSONX_PROJECT_ID")

        if not self.api_key:
            raise ValueError(
                "Watsonx API key required. Set WATSONX_API_KEY environment variable "
                "or pass api_key parameter."
            )
        if not self.project_id:
            raise ValueError(
                "Watsonx project ID required. Set WATSONX_PROJECT_ID environment variable "
                "or pass project_id parameter."
            )

        credentials = Credentials(
            url=base_url,
            api_key=self.api_key,
        )

        self.model = ModelInference(
            model_id=model,
            credentials=credentials,
            project_id=self.project_id,
        )

    def generate(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> str:
        """Generate response using Watsonx."""
        # Combine messages into prompt
        prompt = "\n\n".join(
            f"{msg.role.upper()}: {msg.content}" for msg in messages
        )

        params = {
            "temperature": temperature,
            "max_new_tokens": max_tokens,
        }

        response = self.model.generate_text(prompt=prompt, params=params)
        return response


class OllamaLLM(LLMBackend):
    """Ollama (local) backend."""

    def __init__(
        self,
        model: str = "llama3",
        base_url: str = "http://localhost:11434",
    ):
        """Initialize Ollama backend.

        Parameters
        ----------
        model:
            Model name (e.g., 'llama3', 'mistral', 'phi3').
        base_url:
            Ollama server URL.
        """
        import requests

        self.model = model
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def generate(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> str:
        """Generate response using Ollama."""
        # Convert messages to Ollama format
        formatted_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": formatted_messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        response = self.session.post(url, json=payload, timeout=60)
        response.raise_for_status()

        data = response.json()
        return data.get("message", {}).get("content", "")


def get_llm_backend(
    backend: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> LLMBackend:
    """Factory function to get LLM backend.

    Parameters
    ----------
    backend:
        Backend type: 'mock', 'claude', 'openai', 'watsonx', 'ollama'.
        If None, uses provider from settings.
    api_key:
        API key for the backend (if required).
    model:
        Model name (if applicable).

    Returns
    -------
    llm:
        LLM backend instance.

    Examples
    --------
    >>> # Use mock (no API required)
    >>> llm = get_llm_backend("mock")
    >>> # Use Claude with API key
    >>> llm = get_llm_backend("claude", api_key="sk-...")
    >>> # Use settings
    >>> llm = get_llm_backend()  # Uses provider from settings
    """
    # If no backend specified, use settings
    if backend is None:
        settings = get_settings()
        backend = settings.provider.value

    if backend == "mock":
        return MockLLM()

    elif backend == "claude":
        settings = get_settings()
        api_key = api_key or settings.claude.api_key
        model = model or settings.claude.model
        return ClaudeLLM(api_key=api_key, model=model)

    elif backend == "openai":
        settings = get_settings()
        api_key = api_key or settings.openai.api_key
        model = model or settings.openai.model
        return OpenAILLM(api_key=api_key, model=model)

    elif backend == "watsonx":
        settings = get_settings()
        api_key = api_key or settings.watsonx.api_key
        model = model or settings.watsonx.model_id
        return WatsonxLLM(
            api_key=api_key,
            project_id=settings.watsonx.project_id,
            model=model,
            base_url=settings.watsonx.base_url,
        )

    elif backend == "ollama":
        settings = get_settings()
        model = model or settings.ollama.model
        return OllamaLLM(model=model, base_url=settings.ollama.base_url)

    else:
        raise ValueError(
            f"Unknown backend: {backend}. "
            "Choose 'mock', 'claude', 'openai', 'watsonx', or 'ollama'."
        )


def build_crewai_llm() -> Any:
    """Build a CrewAI-compatible LLM using the active provider.

    Returns
    -------
    llm:
        CrewAI LLM instance configured with active provider.

    Raises
    ------
    ImportError:
        If crewai package is not installed.

    Examples
    --------
    >>> from physica.cognitive import build_crewai_llm
    >>> llm = build_crewai_llm()
    >>> # Use with CrewAI agents
    >>> from crewai import Agent
    >>> agent = Agent(role="Physicist", llm=llm, ...)
    """
    try:
        from crewai import LLM
    except ImportError:
        raise ImportError(
            "crewai package required for CrewAI integration. "
            "Install with: pip install crewai"
        )

    settings = get_settings()
    provider = settings.provider

    if provider == LLMProvider.mock:
        raise ValueError(
            "Mock LLM cannot be used with CrewAI. "
            "Please configure a real provider (openai, claude, watsonx, ollama)."
        )

    elif provider == LLMProvider.openai:
        api_key = settings.openai.api_key or os.getenv("OPENAI_API_KEY", "")
        model = settings.openai.model or "gpt-4o-mini"
        base_url = settings.openai.base_url or os.getenv("OPENAI_BASE_URL", "")

        if not api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable."
            )

        # Ensure model has provider prefix
        if not model.startswith("openai/"):
            model = f"openai/{model}"

        return LLM(
            model=model,
            api_key=api_key,
            base_url=base_url if base_url else None,
        )

    elif provider == LLMProvider.claude:
        api_key = settings.claude.api_key or os.getenv("ANTHROPIC_API_KEY", "")
        model = settings.claude.model or "claude-sonnet-4-5"
        base_url = settings.claude.base_url or os.getenv("ANTHROPIC_BASE_URL", "")

        if not api_key:
            raise ValueError(
                "Claude API key required. Set ANTHROPIC_API_KEY environment variable."
            )

        # CRITICAL: Set as environment variable for CrewAI
        os.environ["ANTHROPIC_API_KEY"] = api_key
        if base_url:
            os.environ["ANTHROPIC_BASE_URL"] = base_url

        # Ensure model has provider prefix
        if not model.startswith("anthropic/"):
            model = f"anthropic/{model}"

        return LLM(
            model=model,
            api_key=api_key,
            base_url=base_url if base_url else None,
        )

    elif provider == LLMProvider.watsonx:
        api_key = settings.watsonx.api_key or os.getenv("WATSONX_API_KEY", "")
        project_id = settings.watsonx.project_id or os.getenv("WATSONX_PROJECT_ID", "")
        model = settings.watsonx.model_id or "ibm/granite-3-8b-instruct"
        base_url = settings.watsonx.base_url or "https://us-south.ml.cloud.ibm.com"

        if not api_key or not project_id:
            raise ValueError(
                "Watsonx API key and project ID required. "
                "Set WATSONX_API_KEY and WATSONX_PROJECT_ID environment variables."
            )

        # CRITICAL: Set environment variables for watsonx
        os.environ["WATSONX_PROJECT_ID"] = project_id
        os.environ["WATSONX_URL"] = base_url

        # Ensure model has provider prefix
        if not model.startswith("watsonx/"):
            model = f"watsonx/{model}"

        return LLM(
            model=model,
            api_key=api_key,
            base_url=base_url,
            project_id=project_id,
            temperature=0.3,
            max_tokens=1024,
        )

    elif provider == LLMProvider.ollama:
        model = settings.ollama.model or "llama3"
        base_url = settings.ollama.base_url or "http://localhost:11434"

        if not base_url:
            raise ValueError(
                "Ollama base URL required. Set OLLAMA_BASE_URL environment variable."
            )

        # Ensure model has provider prefix
        if not model.startswith("ollama/"):
            model = f"ollama/{model}"

        return LLM(model=model, base_url=base_url)

    else:
        raise ValueError(f"Unsupported provider: {provider}")
