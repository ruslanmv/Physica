"""Cognitive Layer: LLM-powered intent parsing and reasoning.

This module provides the interface between natural language intent and
formal physical parameters. Supports multiple LLM providers with CrewAI compatibility.
"""

from .catalog import (
    get_available_models,
    list_models_for_provider,
)
from .intent import IntentParser, PhysicsIntent, PlanningAgent
from .llm import (
    ClaudeLLM,
    LLMBackend,
    MockLLM,
    OllamaLLM,
    OpenAILLM,
    WatsonxLLM,
    build_crewai_llm,
    get_llm_backend,
)
from .settings import (
    LLMProvider,
    PhysicaSettings,
    get_settings,
    set_provider,
    update_settings,
)

__all__ = [
    # Intent parsing
    "IntentParser",
    "PhysicsIntent",
    "PlanningAgent",
    # LLM backends
    "LLMBackend",
    "MockLLM",
    "OpenAILLM",
    "ClaudeLLM",
    "WatsonxLLM",
    "OllamaLLM",
    "get_llm_backend",
    "build_crewai_llm",
    # Settings
    "PhysicaSettings",
    "LLMProvider",
    "get_settings",
    "set_provider",
    "update_settings",
    # Model catalog
    "list_models_for_provider",
    "get_available_models",
]
