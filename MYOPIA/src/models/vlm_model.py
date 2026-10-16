import os
from openai import OpenAI

class ModelClient:
    """Handles model API clients and configurations."""

    # Every endpoint is read from the environment. The two hosted providers ship a
    # default so a bare checkout still runs; Ollama is self-hosted and has none, so
    # OLLAMA_BASE_URL must be set explicitly.
    MODEL_CONFIGS = {
        'gpt-4o': {
            'provider': 'openrouter',
            'api_key_env': 'OPENROUTER_API_KEY',
            'base_url_env': 'OPENROUTER_BASE_URL',
            'default_base_url': 'https://openrouter.ai/api/v1',
        },
        'qwen-vl-max': {
            'provider': 'dashscope',
            'api_key_env': 'ALIBABA_API_KEY',
            'base_url_env': 'ALIBABA_BASE_URL',
            'default_base_url': 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1',
        },
        'llama3.2-vision:11b-instruct-fp16': {
            'provider': 'ollama',
            'base_url_env': 'OLLAMA_BASE_URL',
        },
    }

    def __init__(self, model_name: str):
        if model_name not in self.MODEL_CONFIGS:
            raise ValueError(f"Unsupported model: {model_name}")

        self.model_name = model_name
        self.config = self.MODEL_CONFIGS[model_name]
        self.provider = self.config['provider']
        self.client = self._create_client()

    @property
    def uses_openrouter(self) -> bool:
        """Whether OpenRouter-only request extras apply to this client.

        Provider pinning (extra_body.provider) is an OpenRouter routing feature, so
        it is keyed on the provider rather than on the model name: DashScope and
        Ollama reject or ignore it.
        """
        return self.provider == 'openrouter'

    def _create_client(self) -> OpenAI:
        """Create OpenAI client for the specified model."""
        config = self.config

        base_url = os.getenv(config['base_url_env']) or config.get('default_base_url')
        if not base_url:
            raise ValueError(
                f"{config['base_url_env']} not set in the environment for {self.model_name}"
            )

        if 'api_key_env' in config:
            api_key = os.getenv(config['api_key_env'])
            if not api_key:
                raise ValueError(f"API key not found for {self.model_name}")
        else:
            api_key = "ollama"  # unused by Ollama, OpenAI client just requires a non-empty string

        return OpenAI(
            api_key=api_key,
            base_url=base_url
        )
