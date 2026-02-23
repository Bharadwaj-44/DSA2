import os
import openai
import json
from programmer import Programmer
from inspector import Inspector
from cache.cache import *
from prompt_engineering.prompts import *
import warnings
import traceback
import zipfile
from kernel import *
from display import *
from pathlib import Path
from utils.utils import *
import tiktoken
import time  # CHANGE 1: Added time import for timing logs
from sfassist_client import SFAssistClient
#from horizon_client import SFAssistClient
# warnings.filterwarnings("ignore")
 
 
class Conversation:
 
    def __init__(self, config) -> None:
        self.config = config    
        # Use Snowflake Cortex
        from sfassist_client import SFAssistClient
        self.client = SFAssistClient(config)
        self.is_anthropic = False
        self.model = config['conv_model']
        self.programmer = Programmer(api_key=config['api_key'], model=config['programmer_model'],
                                     base_url=config['base_url_programmer'], config=config)
        self.inspector = Inspector(api_key=config['api_key'], model=config['inspector_model'],
                                   base_url=config['base_url_inspector'], config=config)
        self.session_cache_path = config["session_cache_path"]
        self.chat_history_display = config["chat_history_display"] if "chat_history_display" in config else []
        self.retrieval = self.config['retrieval']
       
        # ============================================================
        # CHANGE 2: Added kernel creation diagnostic logs
        # ============================================================
        print(f" [KERNEL] Creating kernel at: {time.strftime('%H:%M:%S')}")
        print(f" [KERNEL] Session cache path: {self.session_cache_path}")
       
        try:
            self.kernel = CodeKernel(session_cache_path=self.session_cache_path, max_exe_time=config['max_exe_time'])
            print(f" [KERNEL] Kernel created successfully at: {time.strftime('%H:%M:%S')}")
            print(f" [KERNEL] Kernel alive: {self.kernel.is_alive()}")
        except Exception as e:
            print(f" [KERNEL] Kernel creation FAILED: {e}")
            traceback.print_exc()
            raise
       
        self.max_attempts = config['max_attempts']
        self.error_count = 0
        self.repair_count = 0
        self.file_list = []
        self.figure_list = config["figure_list"] if "figure_list" in config else []
        self.function_repository = {}
        self.my_data_cache = None
        self.max_context_tokens = config.get('max_context_tokens', 7000)  # Default to 7000 tokens
        self.encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding
        
        # FIX 2b REVISED: Store original system prompt to prevent growth
        self._original_system_prompt = None
       
        # ============================================================
        # CHANGE 3: Added timing log for initial imports
        # ============================================================
        print(f" [KERNEL] Running initial imports at: {time.strftime('%H:%M:%S')}")
        self.run_code(IMPORT)
        print(f" [KERNEL] Initial imports completed at: {time.strftime('%H:%M:%S')}")
 
 
    def add_functions(self, function_lib: dict) -> None:
        self.function_repository = function_lib
 
    def add_data(self, data_path) -> None:
        self.my_data_cache = data_cache(data_path)
 
    def check_folder(self):
        print(f"DEBUG: check_folder called for path: {self.session_cache_path}")
        current_files = os.listdir(self.session_cache_path)
        print(f"DEBUG: current_files: {current_files}")
        print(f"DEBUG: file_list before: {self.file_list}")
        new_files = set(current_files) - set(self.file_list)
        print(f"DEBUG: new_files: {new_files}")
        self.file_list = current_files
        display = False
        display_link = ''
        if new_files:
            display = True
            for file in new_files:
                file_link = os.path.join(self.session_cache_path,file)
                file_ext = os.path.splitext(file)[1].lower()
               
                if file_ext in ['.png','.jpg','.jpeg']:
                    print(f"DEBUG: Found image file: {file}")
                    display_link += display_image(file_link)
                    absolute_path = Path(file_link).resolve()
                    print("absolute_path",absolute_path)
                    self.figure_list.append(absolute_path)
                elif file_ext not in ['.pkl', '.joblib', '.model']:
                    # Only create download links for non-model files
                    display_link += display_download_file(file_link, file)
        print("display_link:", display_link)
        print(f"DEBUG: check_folder returning display={display}")
        return display, display_link
 
 
    def save_conv(self):
        with open(os.path.join(self.session_cache_path, 'programmer_msg.json'), 'w') as f:
            json.dump(self.programmer.messages, f, indent=4)
            f.close()
        print(f"Conversation saved in {os.path.join(self.session_cache_path, 'programmer_msg.json')}")
        with open(os.path.join(self.session_cache_path, 'inspector_msg.json'), 'w') as f:
            json.dump(self.inspector.messages, f, indent=4)
            f.close()
        print(f"Conversation saved in {os.path.join(self.session_cache_path, 'inspector_msg.json')}")
        with open(os.path.join(self.session_cache_path, 'config.json'), 'w') as f:
            config = {
            "config": self.config,
            "model": self.model,
            "session_cache_path": self.session_cache_path,
            "chat_history_display": self.chat_history_display,
            "retrieval": self.retrieval,
            "max_attempts": self.max_attempts,
            "error_count": self.error_count,
            "repair_count": self.repair_count,
            "file_list": [str(p) for p in self.file_list],
            "figure_list": [str(p) for p in self.figure_list],
            "function_repository": self.function_repository,
        }
            json.dump(config, f, indent=4)
        print(f"Config saved in {os.path.join(self.session_cache_path, 'config.json')}")
 
    def add_programmer_msg(self, message: dict):
        self.programmer.messages.append(message)
    
    def store_original_system_prompt(self):
        """FIX 2b REVISED: Store the original system prompt on first use"""
        if self._original_system_prompt is None and len(self.programmer.messages) > 0:
            if self.programmer.messages[0].get('role') == 'system':
                self._original_system_prompt = self.programmer.messages[0]['content']
                print(f"🔧 FIX 2b: Stored original system prompt ({len(self._original_system_prompt)} chars)")
    
    def update_system_prompt_with_file_context(self, file_context: str):
        """FIX 2b REVISED: Replace system prompt with original + latest file context"""
        self.store_original_system_prompt()
        
        if self._original_system_prompt is not None and len(self.programmer.messages) > 0:
            if self.programmer.messages[0].get('role') == 'system':
                # Replace entire system message content with original + file context
                new_content = self._original_system_prompt + "\n\n" + file_context
                old_length = len(self.programmer.messages[0]['content'])
                self.programmer.messages[0]['content'] = new_content
                new_length = len(new_content)
                print(f"🔧 FIX 2b: System prompt updated - was {old_length} chars, now {new_length} chars")
                return True
        return False
 
    def add_programmer_repair_msg(self, bug_code: str, error_msg: str, fix_method: str, role="user"):
        message = {"role": role,
                   "content": CODE_FIX.format(bug_code=bug_code, error_message=error_msg, fix_method=fix_method)}
        self.programmer.messages.append(message)
 
    def add_inspector_msg(self, bug_code: str, error_msg: str, role="user"):
        message = {"role": role, "content": CODE_INSPECT.format(bug_code=bug_code, error_message=error_msg)}
        self.inspector.messages.append(message)
 
    def run_code(self, code):
        # ============================================================
        # CHANGE 4: Added comprehensive kernel diagnostics before execution
        # ============================================================
        print(f" [KERNEL] run_code START: {time.strftime('%H:%M:%S')}")
        print(f" [KERNEL] Kernel alive before execution: {self.kernel.is_alive()}")
        print(f" [KERNEL] Code length: {len(code)} chars")
        print(f" [KERNEL] Code preview: {code[:200]}...")
       
        try:
            # ============================================================
            # CHANGE 5: Added timing around execute call
            # ============================================================
            exec_start = time.time()
            print(f" [KERNEL] Calling execute() at: {time.strftime('%H:%M:%S')}")
           
            sign, msg_llm, exe_res = execute(code, self.kernel)
           
            exec_end = time.time()
            print(f" [KERNEL] execute() returned at: {time.strftime('%H:%M:%S')}")
            print(f" [KERNEL] Execution took: {exec_end - exec_start:.2f} seconds")
            print(f"DEBUG: run_code returned - sign: {sign}, msg_llm: {msg_llm[:200] if msg_llm else 'None'}...")
           
        except Exception as e:  # this error is due to the outer programme, not the error in the kernel
            print(f' [KERNEL] Error in executing code (outer): {e}')
            print(f' [KERNEL] Kernel alive after error: {self.kernel.is_alive()}')
            traceback.print_exc()
            sign, msg_llm, exe_res = 'text', f'{e}\nThis error is due to the outer programme, not the error in the kernel, you should tell the user to check the system code.', str(e) # tell the user, the code have problems.
 
        # ============================================================
        # CHANGE 6: Added timing log at end of run_code
        # ============================================================
        print(f" [KERNEL] run_code END: {time.strftime('%H:%M:%S')}")
       
        return sign, msg_llm, exe_res
 
    def rendering_code(self):
        for i in range(len(self.programmer.messages) - 1, 0, -1):
            if self.programmer.messages[i]["role"] == "assistant":
                is_python, code = extract_code(self.programmer.messages[i]["content"])
                if is_python:
                    return code
        return None
 
    def show_data(self) -> pd.DataFrame:
        if self.my_data_cache is None:
            print("User do not upload a data.")
            return pd.DataFrame()
        return self.my_data_cache.data
 
    def document_generation(self, chat_history):
        print("Report generating...")
        print(f"DEBUG: Chat history length: {len(chat_history)}")
       
        try:
            formatted_chat = []
            for item in chat_history:
                if len(item) >= 2 and item[0] and item[1]:  # Check if both user and assistant messages exist
                    formatted_chat.append({"role": "user", "content": str(item[0])})
                    formatted_chat.append({"role": "assistant", "content": str(item[1])})
                else:
                    print(f"DEBUG: Skipping malformed chat item: {item}")
           
            print(f"DEBUG: Formatted chat length: {len(formatted_chat)}")
           
            if not formatted_chat:
                print("ERROR: No valid chat history found for report generation")
                # Create a basic report even without chat history
                basic_report = f"""# Data Analysis Report
 
## Dataset Information
No conversation history found. Please ensure you have had a conversation with the AI model before generating a report.
 
## Session Information
- Session Cache Path: {self.session_cache_path}
- Figure List: {self.figure_list}
"""
                mkd_path = os.path.join(self.session_cache_path, 'report.md')
                with open(mkd_path, "w", encoding='utf-8') as f:
                    f.write(basic_report)
                return mkd_path
           
            # Truncate conversation history if too long
            max_tokens = 12000  # Leave room for system prompt and response
            truncated_chat = self._truncate_conversation(formatted_chat, max_tokens)
            print(f"DEBUG: Truncated chat length: {len(truncated_chat)}")
           
            self.messages = [{"role": "system", "content": Basic_Report}] + truncated_chat + [{"role": "user", "content": f"Now, you should generate a report according to the above chat history (Do not give further suggestions at the end of report).\nNote: Here is figure list with links in the chat history: {self.figure_list}"}]
           
            print("DEBUG: Calling chat model for report generation...")
            response = self.call_chat_model()
            if response and hasattr(response, 'choices') and len(response.choices) > 0:
                report = response.choices[0].message.content
                print("DEBUG: Report generated successfully")
            else:
                print("ERROR: Failed to get response from chat model")
                report = "# Report Generation Failed\n\nUnable to generate report due to API error."
           
            self.messages.append({"role": "assistant", "content": report})
            mkd_path = os.path.join(self.session_cache_path, 'report.md')
            with open(mkd_path, "w", encoding='utf-8') as f:
                f.write(report)
            print(f"DEBUG: Report saved to: {mkd_path}")
            return mkd_path
           
        except Exception as e:
            print(f"ERROR in document_generation: {e}")
            import traceback
            traceback.print_exc()
            # Create error report
            error_report = f"""# Report Generation Error
 
An error occurred while generating the report:
 
**Error:** {str(e)}
 
**Chat History Length:** {len(chat_history) if chat_history else 0}
 
Please try again or contact support if the issue persists.
"""
            mkd_path = os.path.join(self.session_cache_path, 'report.md')
            with open(mkd_path, "w", encoding='utf-8') as f:
                f.write(error_report)
            return mkd_path
 
    def _truncate_conversation(self, formatted_chat, max_tokens):
        """Truncate conversation history to fit within token limits"""
        if not formatted_chat:
            return formatted_chat
       
        # Start from the end (most recent messages) and work backwards
        truncated = []
        current_tokens = 0
       
        for i in range(len(formatted_chat) - 1, -1, -1):
            message = formatted_chat[i]
            message_tokens = len(self.encoding.encode(message["content"]))
           
            if current_tokens + message_tokens > max_tokens:
                break
               
            truncated.insert(0, message)
            current_tokens += message_tokens
       
        print(f"DEBUG: Truncated from {len(formatted_chat)} to {len(truncated)} messages")
        print(f"DEBUG: Estimated tokens: {current_tokens}")
       
        return truncated
 
    def export_code(self):
        print("Exporting notebook...")
        notebook_path = os.path.join(self.session_cache_path, 'notebook.ipynb')
        try:
            self.kernel.write_to_notebook(notebook_path)
        except Exception as e:
            print(f"An error occurred when exporting notebook: {e}")
        return notebook_path
 
    def call_chat_model(self, functions=None, include_functions=False):
        # Use OpenAI API format
        params = {
            "model": self.model,
            "messages": self.messages,
        }
 
        if include_functions:
            params["functions"] = functions
            params["function_call"] = "auto"
 
        return self.client.chat.completions.create(**params)
 
 
    def clear(self):
        import shutil
        try:
            # Remove the entire directory and its contents
            if os.path.exists(self.session_cache_path):
                shutil.rmtree(self.session_cache_path)
            # Recreate the directory
            os.makedirs(self.session_cache_path, exist_ok=True)
        except Exception as e:
            print(f"Warning: Could not clear cache directory: {e}")
       
        self.messages = []
        self.programmer.clear()
        self.inspector.clear()
        self.kernel.shutdown()
        del self.kernel
        self.kernel = CodeKernel(session_cache_path=self.session_cache_path, max_exe_time=self.config['max_exe_time'])
        self.my_data_cache = None
 
 
    def stream_workflow(self, chat_history_display, code=None) -> object:
        # ============================================================
        # CHANGE 7: Added timing log at start of stream_workflow
        # ============================================================
        print(f"⏱️ [WORKFLOW] stream_workflow START: {time.strftime('%H:%M:%S')}")
       
        try:
            if chat_history_display and len(chat_history_display) > 0:
                chat_history_display[-1][1] = ""
            yield chat_history_display
            if code is not None:
                prog_response = HUMAN_LOOP.format(code=code)
                self.add_programmer_msg({"role": "user", "content": prog_response})
            else:
                # Manage context before calling the programmer
                self.programmer.messages = self.manage_context(self.programmer.messages, "programmer")
               
                # FIX 3: Add progress indicator before programmer API call
                if chat_history_display and len(chat_history_display) > 0:
                    chat_history_display[-1][1] = "🤔 Analyzing your request..."
                yield chat_history_display
                
                # ============================================================
                # CHANGE 8: Added timing log before API call
                # ============================================================
                print(f" [WORKFLOW] Calling programmer API at: {time.strftime('%H:%M:%S')}")
               
                # Reset the message before streaming starts
                if chat_history_display and len(chat_history_display) > 0:
                    chat_history_display[-1][1] = ""
                
                prog_response = ''
                for message in self.programmer._call_chat_model_streaming(retrieval=self.retrieval, kernel=self.kernel):
                    if chat_history_display and len(chat_history_display) > 0:
                        chat_history_display[-1][1] += message
                    yield chat_history_display
                    prog_response += message
               
                # ============================================================
                # CHANGE 9: Added timing log after API call
                # ============================================================
                print(f" [WORKFLOW] Programmer API returned at: {time.strftime('%H:%M:%S')}")
               
                self.add_programmer_msg({"role": "assistant", "content": prog_response})
 
            is_python, code = extract_code(prog_response)
 
            if is_python:
                if chat_history_display and len(chat_history_display) > 0:
                    chat_history_display[-1][1] += '\n🖥️ Execute code...'
                yield chat_history_display
               
                # ============================================================
                # CHANGE 10: Added timing log before code execution
                # ============================================================
                print(f" [WORKFLOW] About to execute code at: {time.strftime('%H:%M:%S')}")
                print(f" [WORKFLOW] Kernel alive before run_code: {self.kernel.is_alive()}")
               
                sign, msg_llm, exe_res = self.run_code(code)
               
                # ============================================================
                # CHANGE 11: Added timing log after code execution
                # ============================================================
                print(f" [WORKFLOW] Code execution returned at: {time.strftime('%H:%M:%S')}")
                print(f" [WORKFLOW] Sign: {sign}")
               
                if sign and 'error' not in sign:
                    yield from self._handle_execution_result(exe_res, msg_llm, chat_history_display)
                else:
                    self.error_count += 1
                    round = 0
                    while 'error' in sign and round < self.max_attempts:
                        # FIX 3: Add progress indicator for error analysis
                        if chat_history_display and len(chat_history_display) > 0:
                            chat_history_display[-1][1] = f'🔍 Analyzing error, attempt {round + 1}/{self.max_attempts}...'
                        yield chat_history_display
                        
                        self.add_inspector_msg(code, msg_llm)
                        if round == 3:
                            insp_response = "Try other packages or methods."
                        else:
                            # FIX 2d: Trim inspector messages independently (only keep last 3 message pairs)
                            if len(self.inspector.messages) > 7:  # system + 3 pairs (6) = 7 max
                                system_msg = self.inspector.messages[0] if self.inspector.messages[0].get('role') == 'system' else None
                                recent_msgs = self.inspector.messages[-6:]  # Keep last 6 (3 pairs)
                                self.inspector.messages = [system_msg] + recent_msgs if system_msg else recent_msgs
                                print(f"🔧 FIX 2d: Inspector context trimmed to {len(self.inspector.messages)} messages")
                           
                            response = self.inspector._call_chat_model()
                            insp_response = response.choices[0].message.content
                        self.inspector.messages.append({"role": "assistant", "content": insp_response})

                        # FIX 3: Add progress indicator for code repair
                        if chat_history_display and len(chat_history_display) > 0:
                            chat_history_display[-1][1] = f'🔧 Generating fix...'
                        yield chat_history_display

                        self.add_programmer_repair_msg(code, msg_llm, insp_response)
                        
                        # Reset before streaming repair response
                        if chat_history_display and len(chat_history_display) > 0:
                            chat_history_display[-1][1] = ""
                            
                        prog_response = ''
                        for message in self.programmer._call_chat_model_streaming():
                            if chat_history_display and len(chat_history_display) > 0:
                                chat_history_display[-1][1] += message
                            prog_response += message
                            yield chat_history_display
                        if chat_history_display and len(chat_history_display) > 0:
                            chat_history_display[-1][1] += '\n🖥️ Execute code...\n'
                        yield chat_history_display
                        self.add_programmer_msg({"role": "assistant", "content": prog_response})
                        is_python, code = extract_code(prog_response)
                        if is_python:
                            sign, msg_llm, exe_res = self.run_code(code)
                            if sign and 'error' not in sign:
                                self.repair_count += 1
                                break
                        round += 1
 
                    if round == self.max_attempts:
                        return prog_response + f"\nSorry, I can't fix the code with {self.max_attempts} attempts, can you help me to modified it or give some suggestions?"
 
                    yield from self._handle_execution_result(exe_res, msg_llm, chat_history_display)
 
        except Exception as e:
            if chat_history_display and len(chat_history_display) > 0:
                chat_history_display[-1][1] += "\nSorry, there is an error in the program, please try again."
            yield chat_history_display
            print(f"An error occurred: {e}")
            traceback.print_exc()
            if self.programmer.messages[-1]["role"] == "user":
                self.programmer.messages.append({"role": "assistant", "content": f"An error occurred in program: {e}"})
       
        # ============================================================
        # CHANGE 12: Added timing log at end of stream_workflow
        # ============================================================
        print(f" [WORKFLOW] stream_workflow END: {time.strftime('%H:%M:%S')}")
 
    def _handle_execution_result(self, exe_res, msg_llm, chat_history_display):
        if chat_history_display and len(chat_history_display) > 0:
            chat_history_display[-1][1] += display_exe_results(exe_res)
        yield chat_history_display
 
        display, link_info = self.check_folder()
        if chat_history_display and len(chat_history_display) > 0:
            chat_history_display[-1][1] += f"{link_info}" if display else ''
        yield chat_history_display
 
        # FIX 3: Add progress indicator for result analysis
        if chat_history_display and len(chat_history_display) > 0:
            chat_history_display[-1][1] += "\n📊 Analyzing results..."
        yield chat_history_display

        self.add_programmer_msg({"role": "user", "content": RESULT_PROMPT.format(msg_llm)})
        
        # Reset before streaming result analysis
        if chat_history_display and len(chat_history_display) > 0:
            chat_history_display[-1][1] = chat_history_display[-1][1].replace("\n📊 Analyzing results...", "")
            
        prog_response = ''
        for message in self.programmer._call_chat_model_streaming():
            if chat_history_display and len(chat_history_display) > 0:
                chat_history_display[-1][1] += message
            yield chat_history_display
            prog_response += message
 
        self.add_programmer_msg({"role": "assistant", "content": prog_response})
        if chat_history_display and len(chat_history_display) > 0:
            chat_history_display[-1][1] = display_suggestions(prog_response, chat_history_display[-1][1])
        yield chat_history_display
   
   
 
   
    def update_config(self, conv_model, programmer_model, inspector_model, api_key,
                      base_url_conv_model, base_url_programmer, base_url_inspector,
                      max_attempts, max_exe_time):
 
        if self.config['api_key'] != api_key:
            self.config['api_key'] = api_key
            # Update Snowflake clients
            from horizon_client import SFAssistClient
            self.client = SFAssistClient(self.config)
            self.programmer.client = SFAssistClient(self.config)
            self.inspector.client = SFAssistClient(self.config)
 
        if self.model != conv_model:
            self.model = conv_model
            self.config['conv_model'] = conv_model
 
        if self.kernel.max_exe_time != max_exe_time:
            self.kernel.max_exe_time = max_exe_time
            self.config['max_exe_time'] = max_exe_time
 
        if self.programmer.model != programmer_model:
            self.programmer.model = programmer_model
            self.config['programmer_model'] = programmer_model
 
        if self.inspector.model != inspector_model:
            self.inspector.model = inspector_model
            self.config['inspector_model'] = inspector_model
 
        if self.max_attempts != max_attempts:
            self.config['max_attempts'] = max_attempts
 
    def count_tokens(self, text):
        """Count tokens in text using tiktoken"""
        return len(self.encoding.encode(text))
 
    def count_messages_tokens(self, messages):
        """Count total tokens in a list of messages"""
        total_tokens = 0
        for message in messages:
            content = message.get('content', '')
            total_tokens += self.count_tokens(content)
        return total_tokens
 
    def trim_conversation_history(self, messages, max_tokens=None):
 
        """
   
         IMPROVED: Trim conversation history while preserving important context
   
        Always keeps:
   
        1. System message (index 0)
   
        2. Dataset upload messages (critical context)
   
        3. Recent conversation messages
   
        """
   
        if max_tokens is None:
       
            max_tokens = self.max_context_tokens
   
        # Always keep the system message (first message)
   
        if not messages:
       
            return messages
   
        system_message = messages[0] if messages[0].get('role') == 'system' else None
   
        other_messages = messages[1:] if system_message else messages
   
        # Count tokens in system message
   
        system_tokens = self.count_tokens(system_message['content']) if system_message else 0
   
        available_tokens = max_tokens - system_tokens - 500  # Leave buffer for response
   
        #  NEW: Identify and preserve dataset upload messages
   
        dataset_messages = []
   
        regular_messages = []
   
        for msg in other_messages:
       
            content = msg.get('content', '')
   
            # Check if this is a dataset upload message
   
            if 'Dataset Upload Notification' in content or 'Dataset Information' in content:
           
                dataset_messages.append(msg)
   
            else:
           
                regular_messages.append(msg)
   
        # Keep recent messages that fit within token limit
   
        trimmed_messages = []
   
        current_tokens = 0
   
        # Start from the most recent regular messages and work backwards
   
        for message in reversed(regular_messages):
       
            message_tokens = self.count_tokens(message['content'])
   
            if current_tokens + message_tokens <= available_tokens:
           
                trimmed_messages.insert(0, message)
   
                current_tokens += message_tokens
   
            else:
           
                break
           
        #  NEW: Try to include dataset messages if space allows
   
        for dataset_msg in dataset_messages:
       
            dataset_tokens = self.count_tokens(dataset_msg['content'])
   
            if current_tokens + dataset_tokens <= available_tokens:
           
                # Insert dataset messages near the beginning (after system)
   
                trimmed_messages.insert(0, dataset_msg)
   
                current_tokens += dataset_tokens
   
        # Reconstruct the message list
   
        result = [system_message] + trimmed_messages if system_message else trimmed_messages
   
        print(f"Context management: Trimmed from {len(messages)} to {len(result)} messages")
   
        print(f"Token usage: {system_tokens + current_tokens}/{max_tokens}")
   
        print(f"Preserved dataset messages: {len([m for m in result if 'Dataset Upload' in m.get('content', '')])}")
   
        return result
 
    def compress_old_messages(self, messages, max_tokens=None):
        """Compress old messages into summaries to save context"""
        if max_tokens is None:
            max_tokens = self.max_context_tokens
           
        if not messages:
            return messages
           
        # Always keep system message and recent messages
        system_message = messages[0] if messages[0].get('role') == 'system' else None
        other_messages = messages[1:] if system_message else messages
       
        if len(other_messages) <= 10:  # Don't compress if we have few messages
            return messages
           
        # Keep the last 5 messages as-is, compress the rest
        recent_messages = other_messages[-5:]
        old_messages = other_messages[:-5]
       
        # Create a summary of old messages
        if old_messages:
            summary_content = "Previous conversation summary:\n"
            for i, msg in enumerate(old_messages):
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')[:200]  # Truncate long content
                summary_content += f"{i+1}. {role}: {content}...\n"
           
            # Create summary message
            summary_message = {
                "role": "system",
                "content": summary_content
            }
           
            # Reconstruct with summary + recent messages
            result = [system_message, summary_message] + recent_messages if system_message else [summary_message] + recent_messages
        else:
            result = messages
           
        print(f"Context compression: Compressed {len(old_messages)} old messages into summary")
        return result
 
    def manage_context(self, messages, agent_type="programmer"):
        """
        REVISED: Message-count PRIMARY, token-based BACKUP, no compression
        
        1. Use message-count-based approach as PRIMARY (MAX_MESSAGES = 20)
        2. Keep token-based trimming as BACKUP (for huge individual messages)
        3. Remove compression step - no fake "summary" system messages
        4. Always preserve: system message + dataset messages + last N regular messages
        """
        if not messages:
            return messages
           
        # PRIMARY: Message-count-based approach (MAX_MESSAGES = 20)
        MAX_MESSAGES = 20
        
        # Always preserve system message (index 0)
        system_message = messages[0] if messages and messages[0].get('role') == 'system' else None
        other_messages = messages[1:] if system_message else messages
        
        # Separate dataset messages (critical context) from regular messages
        dataset_messages = []
        regular_messages = []
        
        for msg in other_messages:
            content = msg.get('content', '')
            if 'Dataset Upload Notification' in content or 'Dataset Information' in content:
                dataset_messages.append(msg)
            else:
                regular_messages.append(msg)
        
        # PRIMARY TRIMMING: Keep only last N regular messages to stay under MAX_MESSAGES
        available_slots = MAX_MESSAGES - (1 if system_message else 0) - len(dataset_messages)
        if len(regular_messages) > available_slots:
            regular_messages = regular_messages[-available_slots:]
            print(f"🔧 PRIMARY: Message-count limit applied - kept last {available_slots} regular messages")
        
        # Reconstruct: system + dataset + recent regular messages
        result = []
        if system_message:
            result.append(system_message)
        result.extend(dataset_messages)
        result.extend(regular_messages)
        
        print(f"📊 Context managed: {len(messages)} → {len(result)} messages")
        print(f"   • System: {1 if system_message else 0}")
        print(f"   • Dataset: {len(dataset_messages)}")
        print(f"   • Regular: {len(regular_messages)}")
        
        # BACKUP: Token-based trimming only for huge individual messages
        current_tokens = self.count_messages_tokens(result)
        if current_tokens > self.max_context_tokens:
            print(f"⚠️ BACKUP: Over token limit: {current_tokens} > {self.max_context_tokens}")
            print("   Applying token-based trimming for huge individual messages...")
            result = self.trim_conversation_history(result, self.max_context_tokens - 1000)
            
        return result
