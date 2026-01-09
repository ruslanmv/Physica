"""Intent parsing and planning using LLMs.

Translates natural language objectives into formal physical parameters.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .llm import LLMBackend, LLMMessage, MockLLM


class PhysicsIntent(BaseModel):
    """Structured representation of physics intent.

    This is the bridge between natural language and formal physics parameters.
    """

    intent: str = Field(description="High-level intent (e.g., 'simulate_projectile', 'optimize_trajectory')")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Physical parameters")
    objectives: List[str] = Field(default_factory=list, description="Optimization objectives")
    constraints: List[str] = Field(default_factory=list, description="Physical constraints to enforce")
    domain: str = Field(default="mechanics", description="Physics domain (mechanics, thermodynamics, EM)")

    def to_json(self) -> str:
        """Convert to JSON string."""
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> PhysicsIntent:
        """Parse from JSON string."""
        return cls.model_validate_json(json_str)


class IntentParser:
    """Parse natural language into structured physics intents.

    Uses LLMs to understand user objectives and translate them into
    formal physical parameters and constraints.
    """

    def __init__(self, llm: Optional[LLMBackend] = None):
        """Initialize intent parser.

        Parameters
        ----------
        llm:
            LLM backend to use. If None, uses MockLLM.
        """
        self.llm = llm or MockLLM()

        self.system_prompt = """You are a physics AI assistant that translates natural language
physics problems into structured JSON representations.

Your task is to parse user requests and output a JSON object with:
- "intent": The main objective (e.g., "simulate_projectile", "optimize_trajectory", "analyze_heat_transfer")
- "parameters": Physical parameters as a dictionary (e.g., {"velocity": 50.0, "angle": 45.0})
- "objectives": List of optimization goals (e.g., ["maximize_range", "minimize_energy"])
- "constraints": Physical laws to enforce (e.g., ["energy_conservation", "momentum_conservation"])
- "domain": Physics domain ("mechanics", "thermodynamics", "electromagnetism", "quantum")

Be precise with numerical values. If values are not specified, use reasonable defaults.
Always enforce fundamental conservation laws as constraints.

Output ONLY valid JSON, no additional text."""

    def parse(self, user_input: str) -> PhysicsIntent:
        """Parse natural language input into structured intent.

        Parameters
        ----------
        user_input:
            Natural language description of physics problem.

        Returns
        -------
        intent:
            Structured physics intent.
        """
        messages = [
            LLMMessage(role="system", content=self.system_prompt),
            LLMMessage(role="user", content=user_input),
        ]

        response = self.llm.generate(messages, temperature=0.3)

        # Extract JSON from response (handle markdown code blocks)
        response = response.strip()
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            response = response.split("```")[1].split("```")[0].strip()

        try:
            intent_dict = json.loads(response)
            return PhysicsIntent(**intent_dict)
        except Exception as e:
            raise ValueError(f"Failed to parse LLM response as JSON: {e}\nResponse: {response}")


@dataclass
class PhysicsPlan:
    """A multi-step plan for solving a physics problem.

    Attributes
    ----------
    steps:
        Ordered list of intents to execute.
    reasoning:
        Natural language explanation of the plan.
    estimated_complexity:
        Rough estimate of computational cost.
    """

    steps: List[PhysicsIntent] = field(default_factory=list)
    reasoning: str = ""
    estimated_complexity: str = "low"


class PlanningAgent:
    """High-level planning agent that decomposes complex problems.

    Uses LLMs to:
    1. Understand complex multi-step physics problems
    2. Break them into manageable sub-problems
    3. Reason about causal relationships
    4. Generate execution plans
    """

    def __init__(self, llm: Optional[LLMBackend] = None):
        """Initialize planning agent.

        Parameters
        ----------
        llm:
            LLM backend for reasoning.
        """
        self.llm = llm or MockLLM()
        self.intent_parser = IntentParser(llm=self.llm)

        self.planning_prompt = """You are an expert physics planning agent.

Given a complex physics problem, your task is to:
1. Break it down into concrete, sequential steps
2. Identify what simulations or calculations are needed
3. Specify the order of operations
4. Identify dependencies between steps

For each step, describe:
- What physics simulation or calculation to perform
- What parameters are needed
- What constraints must be satisfied
- How this step feeds into the next

Output your plan as a JSON list of step descriptions.
Be specific and actionable."""

    def plan(self, user_request: str) -> PhysicsPlan:
        """Create a multi-step plan for a complex problem.

        Parameters
        ----------
        user_request:
            Complex physics problem description.

        Returns
        -------
        plan:
            Multi-step execution plan.
        """
        # Get high-level plan from LLM
        messages = [
            LLMMessage(role="system", content=self.planning_prompt),
            LLMMessage(role="user", content=user_request),
        ]

        response = self.llm.generate(messages, temperature=0.5, max_tokens=1500)

        # For now, parse as single intent (can be extended to multi-step)
        # In a full implementation, this would parse multiple steps
        intent = self.intent_parser.parse(user_request)

        plan = PhysicsPlan(
            steps=[intent],
            reasoning=response,
            estimated_complexity="medium",
        )

        return plan

    def refine_with_feedback(
        self,
        plan: PhysicsPlan,
        feedback: str,
    ) -> PhysicsPlan:
        """Refine a plan based on execution feedback.

        This enables self-correction based on physical constraints.

        Parameters
        ----------
        plan:
            Original plan.
        feedback:
            Feedback from execution (e.g., conservation law violations).

        Returns
        -------
        refined_plan:
            Updated plan addressing the feedback.
        """
        refinement_prompt = f"""The following physics plan was executed but encountered issues:

Original Plan:
{plan.reasoning}

Feedback:
{feedback}

Please refine the plan to address the issues. Adjust parameters, add constraints,
or change the approach as needed to satisfy physical laws."""

        messages = [
            LLMMessage(role="system", content=self.planning_prompt),
            LLMMessage(role="user", content=refinement_prompt),
        ]

        response = self.llm.generate(messages, temperature=0.5)

        # Create refined plan
        refined = PhysicsPlan(
            steps=plan.steps,  # In full version, would re-parse steps
            reasoning=response,
            estimated_complexity=plan.estimated_complexity,
        )

        return refined
