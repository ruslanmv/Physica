"""Demo: Multi-Provider LLM Support with Model Catalog.

Demonstrates how Physica supports multiple LLM providers:
- OpenAI (GPT-4, GPT-4o-mini, etc.)
- Claude (Anthropic)
- Watsonx (IBM)
- Ollama (local models)
- Mock (no API required for demos)
"""

from physica.cognitive import (
    LLMProvider,
    get_available_models,
    get_llm_backend,
    get_settings,
)


def main():
    print("=" * 80)
    print("MULTI-PROVIDER LLM SUPPORT DEMONSTRATION")
    print("=" * 80)
    print()

    # Show current settings
    print("Current Settings:")
    print("─" * 80)
    settings = get_settings()
    print(f"  Active Provider: {settings.provider.value}")
    print(f"  OpenAI Model: {settings.openai.model}")
    print(f"  Claude Model: {settings.claude.model}")
    print(f"  Watsonx Model: {settings.watsonx.model_id}")
    print(f"  Ollama Model: {settings.ollama.model}")
    print()

    # List available models from catalog
    print("=" * 80)
    print("MODEL CATALOG")
    print("=" * 80)
    print()

    print("Fetching available models from providers...")
    print()

    catalog = get_available_models()

    for provider, models in catalog.items():
        print(f"{provider.upper()}:")
        if models:
            for model in models[:5]:  # Show first 5
                print(f"  • {model}")
            if len(models) > 5:
                print(f"  ... and {len(models) - 5} more")
        else:
            print("  (No models available or API key not configured)")
        print()

    # Demo: Using different providers
    print("=" * 80)
    print("USING DIFFERENT PROVIDERS")
    print("=" * 80)
    print()

    # Example 1: Mock LLM (no API required)
    print("1. Using Mock LLM (no API required)")
    print("─" * 80)
    llm = get_llm_backend("mock")
    from physica.cognitive.llm import LLMMessage

    response = llm.generate([
        LLMMessage(role="user", content="Simulate projectile at 45 degrees")
    ])
    print(f"Response: {response[:100]}...")
    print()

    # Example 2: Using settings
    print("2. Using LLM from Settings")
    print("─" * 80)
    print(f"Current provider: {settings.provider.value}")
    llm = get_llm_backend()  # Uses provider from settings
    print(f"✓ Created {llm.__class__.__name__} backend")
    print()

    # Example 3: Switching providers
    print("3. Switching Providers")
    print("─" * 80)

    available_providers = [
        LLMProvider.mock,
        LLMProvider.openai,
        LLMProvider.claude,
        LLMProvider.watsonx,
        LLMProvider.ollama,
    ]

    for provider in available_providers:
        print(f"  {provider.value}: ", end="")
        try:
            # Don't actually switch, just show it's possible
            if provider == LLMProvider.mock:
                llm = get_llm_backend(provider.value)
                print(f"✓ {llm.__class__.__name__}")
            else:
                print("available (requires API key/config)")
        except Exception as e:
            print(f"✗ {str(e)[:50]}")

    print()

    # Example 4: Provider-specific features
    print("=" * 80)
    print("PROVIDER-SPECIFIC FEATURES")
    print("=" * 80)
    print()

    print("OpenAI:")
    print("  • Models: GPT-4o, GPT-4-turbo, GPT-3.5-turbo")
    print("  • Azure OpenAI support via base_url")
    print()

    print("Claude (Anthropic):")
    print("  • Models: Claude 3.5 Sonnet, Claude 3 Opus")
    print("  • Best-in-class reasoning")
    print()

    print("Watsonx (IBM):")
    print("  • Models: Llama 3, Granite, Mixtral")
    print("  • Enterprise-grade with IBM Cloud")
    print("  • Multi-region support")
    print()

    print("Ollama:")
    print("  • Local models: Llama 3, Mistral, Phi-3")
    print("  • Runs on your hardware")
    print("  • No API keys required")
    print()

    # Configuration example
    print("=" * 80)
    print("CONFIGURATION EXAMPLE")
    print("=" * 80)
    print()

    print("To configure a provider, set environment variables:")
    print()
    print("# OpenAI")
    print("export OPENAI_API_KEY='sk-...'")
    print("export PHYSICA_OPENAI_MODEL='gpt-4o-mini'")
    print()
    print("# Claude")
    print("export ANTHROPIC_API_KEY='sk-ant-...'")
    print("export PHYSICA_CLAUDE_MODEL='claude-sonnet-4-5'")
    print()
    print("# Watsonx")
    print("export WATSONX_API_KEY='...'")
    print("export WATSONX_PROJECT_ID='...'")
    print("export PHYSICA_WATSONX_MODEL='meta-llama/llama-3-3-70b-instruct'")
    print()
    print("# Ollama (local)")
    print("export OLLAMA_BASE_URL='http://localhost:11434'")
    print("export PHYSICA_OLLAMA_MODEL='llama3'")
    print()
    print("# Set active provider")
    print("export PHYSICA_PROVIDER='openai'  # or claude, watsonx, ollama, mock")
    print()

    print("=" * 80)
    print("✅ Demo complete!")
    print()
    print("Key Features:")
    print("  • Multiple LLM providers supported")
    print("  • Dynamic model discovery via API")
    print("  • Environment variable configuration")
    print("  • Settings persistence")
    print("  • CrewAI compatibility")
    print()


if __name__ == "__main__":
    main()
