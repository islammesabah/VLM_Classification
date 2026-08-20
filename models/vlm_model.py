import os
from openai import OpenAI

class ModelClient:
    """Handles model API clients and configurations."""
    
    MODEL_CONFIGS = {
        'gpt-4o': {
            'api_key_env': 'OPENROUTER_API_KEY',
            'base_url': 'https://openrouter.ai/api/v1'
        },
        'meta-llama/llama-3.2-11b-vision-instruct': {
            'api_key_env': 'OPENROUTER_API_KEY',
            'base_url': 'https://openrouter.ai/api/v1'
        },
        'meta-llama/llama-4-scout': {
            'api_key_env': 'OPENROUTER_API_KEY',
            'base_url': 'https://openrouter.ai/api/v1'
        },
        'qwen-vl-max': {
            'api_key_env': 'ALIBABA_API_KEY',
            'base_url': 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1'
        },
        # Self-hosted Ollama server, run by us -- serves the same official
        # Meta weights as the (now-dead) OpenRouter LLaMA-3.2-11B-Vision
        # endpoint, at full fp16 precision rather than the 8-bit local
        # fallback used for the GTSRB cross-model check. Server address is
        # env-only (OLLAMA_BASE_URL), never hardcoded here, since it's an
        # internal DFKI host, not something to publish in the repo. Ollama
        # doesn't check the API key, so any non-empty placeholder works.
        'llama3.2-vision:11b-instruct-fp16': {
            'base_url_env': 'OLLAMA_BASE_URL',
        },
    }

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = self._create_client()

    def _create_client(self) -> OpenAI:
        """Create OpenAI client for the specified model."""
        if self.model_name not in self.MODEL_CONFIGS:
            raise ValueError(f"Unsupported model: {self.model_name}")

        config = self.MODEL_CONFIGS[self.model_name]

        if 'base_url_env' in config:
            base_url = os.getenv(config['base_url_env'])
            if not base_url:
                raise ValueError(
                    f"{config['base_url_env']} not set in the environment for {self.model_name}"
                )
            api_key = "ollama"  # unused by Ollama, OpenAI client just requires a non-empty string
        else:
            base_url = config['base_url']
            api_key = os.getenv(config['api_key_env'])
            if not api_key:
                raise ValueError(f"API key not found for {self.model_name}")

        return OpenAI(
            api_key=api_key,
            base_url=base_url
        )
