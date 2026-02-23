import openai
from sfassist_client import SFAssistClient
#from horizon_client import SFAssistClient

class Inspector:

    def __init__(self, api_key, model=None, base_url='', config=None):
        # Use Snowflake Cortex
        #from snowflake_cortex_client import SnowflakeCortexClient
        from sfassist_client import SFAssistClient
        self.client = SFAssistClient(config)
        self.is_snowflake = False
        self.is_anthropic = False
        # Get model from config dynamically
        if model:
            self.model = model
        elif config and 'snowflake' in config:
            self.model = config['snowflake']['model']  # Takes from config.yaml
        elif config:
            self.model = config.get('inspector_model')  # Legacy fallback
        else:
            self.model = 'claude-4-sonnet'  # Only if NO config at all
        self.messages = []
        self.function_repository = {}

    def add_functions(self, function_lib: dict) -> None:
        self.function_repository = function_lib

    def _call_chat_model(self, functions=None, include_functions=False):
        params = {
            "model": self.model,
            "messages": self.messages,
        }

        if include_functions:
            params['functions'] = functions
            params['function_call'] = "auto"

        try:
            # Use OpenAI API format
            return self.client.chat.completions.create(**params)
        except Exception as e:
            print(f"Error calling chat model: {e}")
            return None

    def clear(self):
        self.messages = []
        self.function_repository = {}