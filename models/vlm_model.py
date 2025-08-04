import os
from openai import OpenAI

class ModelClient:
    """Handles model API clients and configurations."""
    
    MODEL_CONFIGS = {
        'gpt-4o': {
            'api_key_env': 'OPENAI_API_KEY',
            'base_url': None
        },
        'meta-llama/llama-3.2-11b-vision-instruct': {
            'api_key_env': 'OPENROUTER_API_KEY',
            'base_url': 'https://openrouter.ai/api/v1'
        },
        'qwen-vl-max': {
            'api_key_env': 'ALIBABA_API_KEY',
            'base_url': 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1'
        }
    }
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = self._create_client()
    
    def _create_client(self) -> OpenAI:
        """Create OpenAI client for the specified model."""
        if self.model_name not in self.MODEL_CONFIGS:
            raise ValueError(f"Unsupported model: {self.model_name}")
        
        config = self.MODEL_CONFIGS[self.model_name]
        api_key = os.getenv(config['api_key_env'])
        
        if not api_key:
            raise ValueError(f"API key not found for {self.model_name}")
        
        return OpenAI(
            api_key=api_key,
            base_url=config['base_url']
        )
