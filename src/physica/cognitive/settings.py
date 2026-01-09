"""Settings management for LLM providers.

Supports multiple providers with environment variable fallbacks and disk persistence.
"""

from __future__ import annotations

import contextlib
import enum
import json
import os
from pathlib import Path

from pydantic import BaseModel, Field

# Configuration directory
CONFIG_DIR = Path.home() / ".physica"
CONFIG_FILE = CONFIG_DIR / "settings.json"


class LLMProvider(str, enum.Enum):
    """Supported LLM providers."""

    mock = "mock"  # No API required (for demos)
    openai = "openai"
    claude = "claude"
    watsonx = "watsonx"
    ollama = "ollama"


class OpenAIConfig(BaseModel):
    """OpenAI provider configuration."""

    api_key: str = Field(default="")
    model: str = Field(default="gpt-4o-mini")
    base_url: str = Field(default="")  # Optional: for Azure OpenAI or proxies


class ClaudeConfig(BaseModel):
    """Claude (Anthropic) provider configuration."""

    api_key: str = Field(default="")
    model: str = Field(default="claude-sonnet-4-5")
    base_url: str = Field(default="")  # Optional: for proxies


class WatsonxConfig(BaseModel):
    """IBM Watsonx.ai provider configuration."""

    api_key: str = Field(default="")
    project_id: str = Field(default="")
    model_id: str = Field(default="meta-llama/llama-3-3-70b-instruct")
    base_url: str = Field(default="https://us-south.ml.cloud.ibm.com")


class OllamaConfig(BaseModel):
    """Ollama (local) provider configuration."""

    base_url: str = Field(default="http://localhost:11434")
    model: str = Field(default="llama3")


class PhysicaSettings(BaseModel):
    """Global settings for Physica LLM integration."""

    provider: LLMProvider = Field(default=LLMProvider.mock)

    # Provider-specific configs
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    claude: ClaudeConfig = Field(default_factory=ClaudeConfig)
    watsonx: WatsonxConfig = Field(default_factory=WatsonxConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)

    @classmethod
    def from_disk(cls) -> PhysicaSettings:
        """Load settings from disk and merge with environment variables.

        Environment variables take precedence over saved settings.
        """
        # Load from disk if exists
        if CONFIG_FILE.exists():
            data = json.loads(CONFIG_FILE.read_text("utf-8"))
            settings = cls.model_validate(data)
        else:
            settings = cls()

        # Override with environment variables
        env_provider = os.getenv("PHYSICA_PROVIDER")
        if env_provider:
            with contextlib.suppress(ValueError):
                settings.provider = LLMProvider(env_provider.lower())

        # OpenAI environment variables
        if os.getenv("OPENAI_API_KEY"):
            settings.openai.api_key = os.getenv("OPENAI_API_KEY")
        if os.getenv("PHYSICA_OPENAI_MODEL"):
            settings.openai.model = os.getenv("PHYSICA_OPENAI_MODEL")
        if os.getenv("OPENAI_BASE_URL"):
            settings.openai.base_url = os.getenv("OPENAI_BASE_URL")

        # Claude environment variables
        if os.getenv("ANTHROPIC_API_KEY"):
            settings.claude.api_key = os.getenv("ANTHROPIC_API_KEY")
        if os.getenv("PHYSICA_CLAUDE_MODEL"):
            settings.claude.model = os.getenv("PHYSICA_CLAUDE_MODEL")
        if os.getenv("ANTHROPIC_BASE_URL"):
            settings.claude.base_url = os.getenv("ANTHROPIC_BASE_URL")

        # Watsonx environment variables
        if os.getenv("WATSONX_API_KEY"):
            settings.watsonx.api_key = os.getenv("WATSONX_API_KEY")
        if os.getenv("WATSONX_PROJECT_ID"):
            settings.watsonx.project_id = os.getenv("WATSONX_PROJECT_ID")
        if os.getenv("PHYSICA_WATSONX_MODEL"):
            settings.watsonx.model_id = os.getenv("PHYSICA_WATSONX_MODEL")
        if os.getenv("WATSONX_BASE_URL"):
            settings.watsonx.base_url = os.getenv("WATSONX_BASE_URL")

        # Ollama environment variables
        if os.getenv("OLLAMA_BASE_URL"):
            settings.ollama.base_url = os.getenv("OLLAMA_BASE_URL")
        if os.getenv("PHYSICA_OLLAMA_MODEL"):
            settings.ollama.model = os.getenv("PHYSICA_OLLAMA_MODEL")

        return settings

    def save(self) -> None:
        """Save settings to disk."""
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_FILE.write_text(self.model_dump_json(indent=2), "utf-8")


# Global settings instance
_settings = PhysicaSettings.from_disk()


def get_settings() -> PhysicaSettings:
    """Get global settings instance."""
    return _settings


def set_provider(provider: LLMProvider) -> PhysicaSettings:
    """Set active LLM provider."""
    _settings.provider = provider
    _settings.save()
    return _settings


def update_settings(updates: dict) -> PhysicaSettings:
    """Update settings with partial or full configuration.

    Parameters
    ----------
    updates:
        Dictionary with settings updates. Can include:
        - provider: str
        - openai: dict
        - claude: dict
        - watsonx: dict
        - ollama: dict

    Returns
    -------
    settings:
        Updated settings instance.
    """
    global _settings

    # Update provider if present
    if "provider" in updates:
        _settings.provider = LLMProvider(updates["provider"])

    # Update provider-specific configs
    if "openai" in updates:
        _settings.openai = OpenAIConfig(**updates["openai"])
    if "claude" in updates:
        _settings.claude = ClaudeConfig(**updates["claude"])
    if "watsonx" in updates:
        _settings.watsonx = WatsonxConfig(**updates["watsonx"])
    if "ollama" in updates:
        _settings.ollama = OllamaConfig(**updates["ollama"])

    _settings.save()
    return _settings
