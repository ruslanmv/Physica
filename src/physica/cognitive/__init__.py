"""Cognitive Layer: LLM-powered intent parsing and reasoning.

This module provides the interface between natural language intent and
formal physical parameters.
"""

from .intent import IntentParser, PhysicsIntent, PlanningAgent
from .llm import LLMBackend, MockLLM, get_llm_backend

__all__ = [
    "IntentParser",
    "PhysicsIntent",
    "PlanningAgent",
    "LLMBackend",
    "MockLLM",
    "get_llm_backend",
]
