"""Cognitive Layer: LLM-powered intent parsing and reasoning.

This module provides the interface between natural language intent and
formal physical parameters. Supports multiple LLM providers with CrewAI compatibility.
"""

from .intent import IntentParser, PhysicsIntent, PlanningAgent
from .llm import (
    LLMBackend,
    MockLLM,
    OpenAILLM,
    ClaudeLLM,
    WatsonxLLM,
    OllamaLLM,
    get_llm_backend,
    build_crewai_llm,
)
from .settings import (
    PhysicaSettings,
    LLMProvider,
    get_settings,
    set_provider,
    update_settings,
)
from .catalog import (
    list_models_for_provider,
    get_available_models,
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
