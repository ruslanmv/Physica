"""LLM backend abstraction for cognitive layer.

Provides unified interface to different LLM providers (Claude, GPT, local models).
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional


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


def get_llm_backend(
    backend: str = "mock",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> LLMBackend:
    """Factory function to get LLM backend.

    Parameters
    ----------
    backend:
        Backend type: 'mock', 'claude', 'openai'.
    api_key:
        API key for the backend (if required).
    model:
        Model name (if applicable).

    Returns
    -------
    llm:
        LLM backend instance.
    """
    if backend == "mock":
        return MockLLM()
    elif backend == "claude":
        return ClaudeLLM(api_key=api_key, model=model or "claude-3-5-sonnet-20241022")
    elif backend == "openai":
        return OpenAILLM(api_key=api_key, model=model or "gpt-4")
    else:
        raise ValueError(f"Unknown backend: {backend}. Choose 'mock', 'claude', or 'openai'.")
