import openai
from prompt_engineering.prompts import PROGRAMMER_PROMPT
#from knw_in import retrieval_knowledge
#from snowflake_cortex_client import SnowflakeCortexClient
#from horizon_client import SFAssistClient
from sfassist_client import SFAssistClient
import os
import traceback
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Programmer:

    def __init__(self, api_key, model=None, base_url=None, config=None):
        # Use Snowflake Cortex
        self.client = SFAssistClient(config)
        self.is_anthropic = False
        # Get model from config dynamically
        if model:
            self.model = model
        elif config and 'snowflake' in config:
            self.model = config['snowflake']['model']  # Takes from config.yaml
        elif config:
            self.model = config.get('programmer_model')  # Legacy fallback
        else:
            self.model = 'claude-4-sonnet'  # Only if NO config at all
        self.messages = []
        self.function_repository = {}
        self.last_snaps = None

    def add_functions(self, function_lib: dict) -> None:
        self.function_repository = function_lib

    def _call_chat_model(self, functions=None, include_functions=False, retrieval=False):
        if retrieval:
            snaps = retrieval_knowledge(self.messages[-1]["content"])
            if snaps:
                self.last_snaps = snaps
                self.messages[-1]["content"] += snaps
            else:
                self.last_snaps = None

        params = {
            "model": self.model,
            "messages": self.messages,
        }

        if include_functions:
            params['functions'] = functions
            params['function_call'] = "auto"

        try:
            response = self.client.chat.completions.create(**params)
            usage = response.usage
            print(f"======Prompt Tokens: {usage.prompt_tokens}======Completion Tokens: {usage.completion_tokens}=======Total Tokens: {usage.total_tokens}")
            return response
        except Exception as e:
            print(f"Error calling chat model: {e}")
            return None

    def _call_chat_model_streaming(self, functions=None, include_functions=False, retrieval=False, kernel=None):
        print(f"DEBUG: _call_chat_model_streaming called with model={self.model}")
        print(f"DEBUG: Messages: {self.messages}")
        temp = self.messages[-1]["content"]
        if retrieval:
            snaps = retrieval_knowledge(self.messages[-1]["content"], kernel=kernel)
            if snaps:
                for chunk in snaps:
                    yield chunk
                self.last_snaps = snaps
                self.messages[-1]["content"] += snaps
            else:
                self.last_snaps = None

        params = {
            "model": self.model,
            "messages": self.messages,
            "max_tokens":8192,
            "stream": True
        }

        if include_functions:
            params['functions'] = functions
            params['function_call'] = "auto"

        print(f"DEBUG: About to call API with params: {params}")
        try:
            # Use OpenAI API format
            stream = self.client.chat.completions.create(**params)
            print("DEBUG: OpenAI API call successful, processing stream")
            self.messages[-1]["content"] = temp
            for chunk in stream:
                if hasattr(chunk, 'choices') and chunk.choices[0].delta.content is not None:
                    chunk_message = chunk.choices[0].delta.content
                    print(f"DEBUG: Received chunk: {chunk_message}")
                    yield chunk_message
        except Exception as e:
            print(f"Error calling chat model: {e}")
            traceback.print_exc()
            return None
    def clear(self):
        self.messages = [
            {
                "role": "system",
                "content": PROGRAMMER_PROMPT
            }
        ]
        self.function_repository = {}




