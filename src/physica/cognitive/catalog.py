"""Model catalog for discovering available models from each provider.

Supports dynamic model fetching from OpenAI, Claude, Watsonx, and Ollama.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import List, Optional, Tuple

import requests

from .settings import LLMProvider, PhysicaSettings, get_settings

# Watsonx.ai configuration
WATSONX_BASE_URLS = [
    "https://us-south.ml.cloud.ibm.com",
    "https://eu-de.ml.cloud.ibm.com",
    "https://jp-tok.ml.cloud.ibm.com",
    "https://au-syd.ml.cloud.ibm.com",
]

WATSONX_ENDPOINT = "/ml/v1/foundation_model_specs"
WATSONX_PARAMS = {
    "version": "2024-09-16",
    "filters": "!function_embedding,!lifecycle_withdrawn",
}

TODAY = datetime.today().strftime("%Y-%m-%d")


def _is_deprecated_or_withdrawn(lifecycle: list) -> bool:
    """Check if a model is deprecated or withdrawn."""
    for entry in lifecycle:
        if entry.get("id") in {"deprecated", "withdrawn"}:
            if entry.get("start_date", "") <= TODAY:
                return True
    return False


def list_openai_models(settings: Optional[PhysicaSettings] = None) -> Tuple[List[str], Optional[str]]:
    """List available OpenAI models.

    Parameters
    ----------
    settings:
        Settings instance. If None, uses global settings.

    Returns
    -------
    models:
        List of model IDs.
    error:
        Error message if failed, None otherwise.
    """
    if settings is None:
        settings = get_settings()

    api_key = settings.openai.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return [], "OpenAI API key not configured"

    base_url = settings.openai.base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com")
    url = f"{base_url.rstrip('/')}/v1/models"

    try:
        resp = requests.get(
            url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])
        models = sorted({m.get("id", "") for m in data if m.get("id")})
        return models, None
    except Exception as e:
        return [], f"Error listing OpenAI models: {e}"


def list_claude_models(settings: Optional[PhysicaSettings] = None) -> Tuple[List[str], Optional[str]]:
    """List available Claude models.

    Parameters
    ----------
    settings:
        Settings instance. If None, uses global settings.

    Returns
    -------
    models:
        List of model IDs.
    error:
        Error message if failed, None otherwise.
    """
    if settings is None:
        settings = get_settings()

    api_key = settings.claude.api_key or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return [], "Claude (Anthropic) API key not configured"

    base_url = settings.claude.base_url or os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
    url = f"{base_url.rstrip('/')}/v1/models"
    anthropic_version = os.getenv("ANTHROPIC_VERSION", "2023-06-01")

    try:
        resp = requests.get(
            url,
            headers={
                "x-api-key": api_key,
                "anthropic-version": anthropic_version,
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])
        models = sorted({m.get("id", "") for m in data if m.get("id")})
        return models, None
    except Exception as e:
        return [], f"Error listing Claude models: {e}"


def list_watsonx_models(settings: Optional[PhysicaSettings] = None) -> Tuple[List[str], Optional[str]]:
    """List available Watsonx foundation models.

    No API key required for listing IBM-managed models.

    Parameters
    ----------
    settings:
        Settings instance. If None, uses global settings.

    Returns
    -------
    models:
        List of model IDs.
    error:
        Error message if failed, None otherwise.
    """
    all_models = set()

    for base in WATSONX_BASE_URLS:
        url = f"{base}{WATSONX_ENDPOINT}"
        try:
            resp = requests.get(url, params=WATSONX_PARAMS, timeout=10)
            resp.raise_for_status()
            resources = resp.json().get("resources", [])

            for m in resources:
                if _is_deprecated_or_withdrawn(m.get("lifecycle", [])):
                    continue
                model_id = m.get("model_id")
                if model_id:
                    all_models.add(model_id)
        except Exception:
            # Skip this region on error
            continue

    if not all_models:
        return [], "No Watsonx models found (public API unavailable?)"

    return sorted(all_models), None


def list_ollama_models(settings: Optional[PhysicaSettings] = None) -> Tuple[List[str], Optional[str]]:
    """List available Ollama models from local/remote server.

    Parameters
    ----------
    settings:
        Settings instance. If None, uses global settings.

    Returns
    -------
    models:
        List of model names.
    error:
        Error message if failed, None otherwise.
    """
    if settings is None:
        settings = get_settings()

    base_url = settings.ollama.base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    url = f"{base_url.rstrip('/')}/api/tags"

    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json().get("models", [])
        models = sorted({m.get("name", "") for m in data if m.get("name")})
        return models, None
    except Exception as e:
        return [], f"Error listing Ollama models from {url}: {e}"


def list_models_for_provider(
    provider: LLMProvider,
    settings: Optional[PhysicaSettings] = None,
) -> Tuple[List[str], Optional[str]]:
    """List available models for a given provider.

    Parameters
    ----------
    provider:
        LLM provider to query.
    settings:
        Settings instance. If None, uses global settings.

    Returns
    -------
    models:
        List of model IDs/names.
    error:
        Error message if failed, None otherwise.

    Examples
    --------
    >>> from physica.cognitive.catalog import list_models_for_provider
    >>> from physica.cognitive.settings import LLMProvider
    >>> models, error = list_models_for_provider(LLMProvider.openai)
    >>> if error is None:
    ...     print(f"Found {len(models)} OpenAI models")
    """
    if provider == LLMProvider.mock:
        return ["mock-model-v1"], None
    elif provider == LLMProvider.openai:
        return list_openai_models(settings)
    elif provider == LLMProvider.claude:
        return list_claude_models(settings)
    elif provider == LLMProvider.watsonx:
        return list_watsonx_models(settings)
    elif provider == LLMProvider.ollama:
        return list_ollama_models(settings)
    else:
        return [], f"Unsupported provider: {provider}"


def get_available_models() -> dict:
    """Get all available models grouped by provider.

    Returns
    -------
    models:
        Dictionary mapping provider names to lists of available models.

    Examples
    --------
    >>> from physica.cognitive.catalog import get_available_models
    >>> catalog = get_available_models()
    >>> print(catalog.keys())
    dict_keys(['openai', 'claude', 'watsonx', 'ollama'])
    """
    settings = get_settings()
    catalog = {}

    for provider in LLMProvider:
        if provider == LLMProvider.mock:
            catalog[provider.value] = ["mock-model-v1"]
        else:
            models, error = list_models_for_provider(provider, settings)
            if error is None:
                catalog[provider.value] = models
            else:
                catalog[provider.value] = []

    return catalog
