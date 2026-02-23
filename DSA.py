import shutil
import gradio as gr
import json
import time
import threading
from conversation import Conversation
from prompt_engineering.prompts import *
import yaml
from utils.utils import *
import sys
import os
#from horizon_client import HorizonClient
 
 
class DSA:
    def __init__(self, config_path='config.yaml'):
        print("Try to load config: ", config_path)
 
        if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
            bundle_dir = os.path.dirname(sys.executable)
        else:
            bundle_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(bundle_dir, config_path)
 
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        if self.config["load_chat"] == True:
            self.load_dialogue(self.config["chat_history_path"])
        else:
            self.session_cache_path = self.init_local_cache_path(to_absolute_path(self.config["project_cache_path"]))
            self.config["session_cache_path"] = self.session_cache_path
        print("Session cache path: ", self.session_cache_path)
        self.conv = Conversation(self.config)

        self.conv.programmer.messages = [
            {
                "role": "system",
                "content": PROGRAMMER_PROMPT.format(working_path=self.session_cache_path)
            }
        ]

        if self.conv.retrieval:
            self.conv.programmer.messages[0]["content"] += KNOWLEDGE_INTEGRATION_SYSTEM
            
        # FIX 2b: Store file context separately to prevent system message growth
        self.file_context = ""
        
        # Cleanup old sessions on startup
        self.cleanup_old_sessions(max_age_hours=12)
        
        # Start background cleanup thread
        self.start_cleanup_thread()
 
 
    def init_local_cache_path(self, project_cache_path):
        current_fold = time.strftime('%Y-%m-%d', time.localtime())
        hsid = str(hash(id(self)))  # new_uuid = str(uuid.uuid4())
        session_cache_path = os.path.join(project_cache_path, current_fold + '-' + hsid)
        if not os.path.exists(session_cache_path):
            os.makedirs(session_cache_path)
        return session_cache_path
 
    def open_board(self):
 
        """
 
         OPTIMIZED: Show dataset info + preview (not full data)
 
        Prevents browser crash on large datasets
 
        """
 
        data = self.conv.show_data()
 
        if data.empty:
 
            print("No data available to display")
 
            return gr.Dataframe(visible=False)
 
        total_rows = len(data)
 
        total_cols = len(data.columns)
 
        print(f"Dataset: {total_rows:,} rows × {total_cols} columns")
 
        #  SAFE: Only show first 100 rows as preview
 
        MAX_PREVIEW_ROWS = 100
 
        if total_rows > MAX_PREVIEW_ROWS:
 
            preview_data = data.head(MAX_PREVIEW_ROWS).copy()
 
            print(f"Showing preview: first {MAX_PREVIEW_ROWS} of {total_rows:,} rows")
 
        else:
 
            preview_data = data
 
            print(f"Showing all {total_rows} rows")
 
        return gr.Dataframe(value=preview_data, visible=True)
 
    def add_file(self, files):
 
        """Add file - COMPANY PRODUCTION VERSION"""
 
        file_path = files.name
 
        shutil.copy(file_path, self.session_cache_path)
 
        filename = os.path.basename(file_path)
 
        # ============================================================
        # CHANGE 1: Use local_cache_path instead of file_path
        # OLD: self.conv.add_data(file_path)
        # NEW: Define local_cache_path FIRST, then use it
        # ============================================================
        local_cache_path = os.path.join(self.session_cache_path, filename)
 
        self.conv.add_data(local_cache_path)  # CHANGE 1: Fixed path
 
        self.conv.file_list.append(filename)
 
        # Get dataset information with truncation for large datasets
 
        try:
 
            gen_info = self.conv.my_data_cache.get_description()
 
            gen_info_str = str(gen_info) if gen_info else "Dataset information not available"
 
            # Truncate if too long (company datasets can be huge)
 
            max_length = 2000
 
            if len(gen_info_str) > max_length:
 
                gen_info_str = gen_info_str[:max_length] + f"\n... (dataset info truncated from {len(gen_info_str)} to {max_length} characters. Full data is still available for analysis.)"
 
                print(f"Dataset info truncated: {len(str(gen_info))} → {max_length} characters")
 
        except Exception as e:
 
            print(f"Warning: Could not get dataset description: {e}")
 
            import traceback
 
            traceback.print_exc()
 
            gen_info_str = "Dataset loaded successfully. You can use pd.read_csv() to analyze it."
 
        #  CRITICAL FIX: Add dataset context as USER/ASSISTANT messages
 
        # NOT as system message append
 
        dataset_message = f"""Dataset Upload Notification:
 
    File uploaded: {filename}
 
    Location: {local_cache_path}
   
    Dataset Information:
 
    {gen_info_str}
   
    Important: Please analyze this dataset carefully, paying attention to missing values and data types of each column."""
   
        # Add as user message (like user is informing the AI)
 
        self.conv.programmer.messages.append({
 
            "role": "user",
 
            "content": dataset_message
 
        })
 
        # Add assistant acknowledgment
 
        self.conv.programmer.messages.append({
 
            "role": "assistant",
 
            "content": f"I've received the dataset '{filename}'. I can see the data structure and am ready to help you analyze it. What would you like to do first?"
 
        })
 
        # Debug logging
 
        print("=" * 80)
 
        print("DATASET CONTEXT ADDED AS CONVERSATION MESSAGES")
 
        print(f"System message size: {len(self.conv.programmer.messages[0]['content'])} chars (unchanged)")
 
        print(f"Total messages in conversation: {len(self.conv.programmer.messages)}")
 
        print("=" * 80)
 
        print(f"Upload file in gradio path: {file_path}, local cache path: {local_cache_path}")
 
    def add_file_with_feedback(self, files):
 
        """Add file with status feedback - FIXED FOR SF ASSIST API + GPT-4"""
 
        if files is None:
 
            return gr.HTML(visible=False)
 
        try:
            # ============================================================
            # CHANGE 2: Added timing log for file upload start
            # ============================================================
            print(f" [TIMING] add_file_with_feedback START: {time.strftime('%H:%M:%S')}")
 
            file_path = files.name
 
            filename = os.path.basename(file_path)
 
            # ============================================================
            # CHANGE 3: Define local_cache_path BEFORE copying and add_data
            # OLD: local_cache_path was defined AFTER add_data call
            # NEW: Define it FIRST, use it in add_data
            # ============================================================
            local_cache_path = os.path.join(self.session_cache_path, filename)
 
            # Copy file to session cache
            shutil.copy(file_path, self.session_cache_path)
           
            # ============================================================
            # CHANGE 4: Verify file was copied successfully
            # ============================================================
            print(f" [TIMING] File copied: {time.strftime('%H:%M:%S')}")
            print(f" [FILE CHECK] Source exists: {os.path.exists(file_path)}")
            print(f" [FILE CHECK] Destination exists: {os.path.exists(local_cache_path)}")
            if os.path.exists(local_cache_path):
                print(f" [FILE CHECK] Destination size: {os.path.getsize(local_cache_path):,} bytes")
 
            # ============================================================
            # CHANGE 5: Use local_cache_path instead of file_path
            # OLD: self.conv.add_data(file_path)
            # NEW: self.conv.add_data(local_cache_path)
            # ============================================================
            self.conv.add_data(local_cache_path)  # CHANGE 5: Fixed path
 
            self.conv.file_list.append(filename)
 
            # Get dataset information with truncation for large datasets
 
            try:
 
                gen_info = self.conv.my_data_cache.get_description()
 
                gen_info_str = str(gen_info) if gen_info else "Dataset information not available"
 
                # Truncate if too long (company datasets can be huge)
 
                max_length = 2000
 
                if len(gen_info_str) > max_length:
 
                    gen_info_str = gen_info_str[:max_length] + f"\n... (dataset info truncated from {len(gen_info_str)} to {max_length} characters. Full data is still available for analysis.)"
 
                    print(f"Dataset info truncated: {len(str(gen_info))} → {max_length} characters")
 
            except Exception as e:
 
                print(f"Warning: Could not get dataset description: {e}")
 
                import traceback
 
                traceback.print_exc()
 
                gen_info_str = "Dataset loaded successfully. You can use pd.read_csv() or pd.read_excel() to analyze it."
 
            # FIX 2b REVISED: Use conversation method to REPLACE file context in system message
            file_context = f"""
   
     CRITICAL - USER UPLOADED FILE 

    Filename: {filename}

    Full Path: {local_cache_path}
   
    IMPORTANT INSTRUCTIONS:

    - When user asks about "the data" or "the dataset", they mean: {filename}

    - DO NOT look for files named 'data.csv'

    - USE THIS EXACT PATH: {local_cache_path}

    """

            # Replace system prompt with original + latest file context
            success = self.conv.update_system_prompt_with_file_context(file_context)
            if success:
                print(" FIX 2b: File context REPLACED in system message (not appended)")
            else:
                print(" FIX 2b: Failed to update system message with file context")
 
            # 2. Then, add as conversation messages (for chat flow)
 
            dataset_message = f"""Dataset Upload Notification:
 
        File uploaded: {filename}
 
        Location: {local_cache_path}
 
        Dataset Information:
 
        {gen_info_str}
 
         IMPORTANT: Use the file at {local_cache_path} for all data operations!"""
 
            # Add as user message (like user is informing the AI)
 
            self.conv.programmer.messages.append({
 
                "role": "user",
 
                "content": dataset_message
 
            })
 
            #  REMOVED: Assistant acknowledgment - causes SF Assist API to lose context
            # self.conv.programmer.messages.append({
            #     "role": "assistant",
            #     "content": f"I've received the dataset '{filename}' located at {local_cache_path}. I can see the data structure and am ready to help you analyze it. What would you like to do first?"
            # })
 
            # Debug logging
 
            print("=" * 80)
 
            print(" FILE CONTEXT ADDED TO SYSTEM MESSAGE + CONVERSATION")
 
            print(f"System message size: {len(self.conv.programmer.messages[0]['content'])} chars")
 
            print(f"File in system message: {filename}")
 
            print(f"Total messages: {len(self.conv.programmer.messages)}")
 
            print("=" * 80)
 
            # Verification debug
 
            print("\n" + "=" * 80)
 
            print(" VERIFICATION: DATASET MESSAGES ADDED")
 
            print("=" * 80)
 
            # Check system message
 
            if filename in self.conv.programmer.messages[0]["content"]:
 
                print(f" System message contains: {filename}")
 
            else:
 
                print(f" WARNING: System message does NOT contain {filename}")
 
            # Find and display the dataset upload messages
 
            for idx, msg in enumerate(self.conv.programmer.messages):
 
                if 'Dataset Upload Notification' in msg.get('content', ''):
 
                    print(f" Found dataset upload message at index {idx}")
 
                    print(f"  Role: {msg['role']}")
 
                    print(f"  Filename present: {'✓' if filename in msg['content'] else '✗'}")
 
                    print(f"  Path present: {'✓' if local_cache_path in msg['content'] else '✗'}")
 
                elif "I've received the dataset" in msg.get('content', ''):
 
                    print(f" Found assistant acknowledgment at index {idx}")
 
            print(f"\n Total messages now: {len(self.conv.programmer.messages)}")
 
            print("=" * 80 + "\n")
           
            #  CHANGE #1: Set flag for file context injection on next question
            self.conv.needs_file_context_injection = True
            print(f" Next user question will have file context injected")
           
            # ============================================================
            # CHANGE 6: Added timing log for file upload end
            # ============================================================
            print(f" [TIMING] add_file_with_feedback END: {time.strftime('%H:%M:%S')}")
            print("=" * 80 + "\n")
 
            # Create success status HTML
 
            status_html = f"""
    <div style="background-color: #d4edda; border: 1px solid #c3e6cb; color: #155724; padding: 10px; border-radius: 5px; margin: 5px 0;">
    <strong> File Uploaded Successfully!</strong><br>
    <strong>File:</strong> {filename}<br>
    <strong>Size:</strong> {os.path.getsize(file_path):,} bytes<br>
    <strong>Type:</strong> {filename.split('.')[-1].upper()}
    </div>
 
            """
 
            print(f"Upload file in gradio path: {file_path}, local cache path: {local_cache_path}")
 
            # Return only the status HTML
 
            return gr.HTML(value=status_html, visible=True)
 
        except Exception as e:
 
            # Create error status HTML
 
            error_html = f"""
    <div style="background-color: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; padding: 10px; border-radius: 5px; margin: 5px 0;">
    <strong> Upload Failed!</strong><br>
    <strong>Error:</strong> {str(e)}
    </div>
 
            """
 
            # Return only the error status HTML
 
            return gr.HTML(value=error_html, visible=True)
   
 
    def rendering_code(self):
        return self.conv.rendering_code()
 
    def generate_report(self, chat_history):
        print(f"DEBUG: generate_report called with chat_history length: {len(chat_history) if chat_history else 0}")
        try:
            down_path = self.conv.document_generation(chat_history)
            print(f"DEBUG: Report generated at path: {down_path}")
            # Convert to absolute path and ensure it exists
            abs_path = os.path.abspath(down_path)
            if os.path.exists(abs_path):
                print(f"DEBUG: File exists at: {abs_path}")
                return [gr.Button(visible=False), gr.DownloadButton(label=f"Download Report", value=abs_path, visible=True)]
            else:
                print(f"ERROR: File not found at: {abs_path}")
                return [gr.Button(visible=True), gr.DownloadButton(visible=False)]
        except Exception as e:
            print(f"ERROR in generate_report: {e}")
            import traceback
            traceback.print_exc()
            # Return error state
            return [gr.Button(visible=True), gr.DownloadButton(visible=False)]
 
    def export_code(self):
        down_path = self.conv.export_code()
        # Convert to absolute path and ensure it exists
        abs_path = os.path.abspath(down_path)
        if os.path.exists(abs_path):
            print(f"DEBUG: Notebook file exists at: {abs_path}")
            return [gr.Button(visible=False), gr.DownloadButton(label=f"Download Notebook", value=abs_path, visible=True)]
        else:
            print(f"ERROR: Notebook file not found at: {abs_path}")
            return [gr.Button(visible=True), gr.DownloadButton(visible=False)]
 
    def down_report(self):
        return [gr.Button(visible=True), gr.DownloadButton(visible=False)]
 
    def down_notebook(self):
        return [gr.Button(visible=True), gr.DownloadButton(visible=False)]
   
   
    def chat_streaming(self, message, chat_history, code=None):
        """
         FIXED: Inject file context on EVERY question
        This ensures Claude always knows about the uploaded file,
        even after context trimming removes old messages.
        """
        # ============================================================
        # CHANGE 7: Added timing logs for chat_streaming
        # ============================================================
        print(f" [TIMING] chat_streaming START: {time.strftime('%H:%M:%S')}")
       
        if not code:
            enhanced_message = message
            #  ALWAYS inject file context if a file is uploaded
            # This is cheap (~30 tokens) and guarantees Claude never forgets the file
            if hasattr(self.conv, 'my_data_cache') and self.conv.my_data_cache is not None:
                # Get file info from data cache
                file_path = self.conv.my_data_cache.file_path
                filename = os.path.basename(file_path)
               
                # ============================================================
                # CHANGE 8: Added file existence check with logging
                # ============================================================
                print(f" [FILE CHECK] Path from cache: {file_path}")
                print(f" [FILE CHECK] File exists: {os.path.exists(file_path)}")
               
                # Compact injection (every question)
                enhanced_message = f"""[Active Dataset: {filename} | Path: {file_path}]
   
    {message}"""
                print(f" Injected file context: {filename}")
            # Fallback: Check file_list if my_data_cache not available
            elif hasattr(self.conv, 'file_list') and len(self.conv.file_list) > 0:
                latest_file = self.conv.file_list[-1]
                file_path = os.path.join(self.session_cache_path, latest_file)
               
                # ============================================================
                # CHANGE 9: Added file existence check for fallback path
                # ============================================================
                print(f" [FILE CHECK] Fallback path: {file_path}")
                print(f" [FILE CHECK] File exists: {os.path.exists(file_path)}")
               
                enhanced_message = f"""[Active Dataset: {latest_file} | Path: {file_path}]
   
    {message}"""
                print(f" Injected file context (fallback): {latest_file}")
           
            self.conv.programmer.messages.append({"role": "user", "content": enhanced_message})
        else:
            message = code
       
        # ============================================================
        # CHANGE 10: Added timing log for chat_streaming end
        # ============================================================
        print(f" [TIMING] chat_streaming END: {time.strftime('%H:%M:%S')}")
       
        return "", chat_history + [[message, None]]
   
    def show_csv_download(self, filename):
        """Show CSV download link when specifically requested"""
        # Check in session cache first
        cache_path = os.path.join(self.session_cache_path, filename)
        if os.path.exists(cache_path):
            return f" **{filename}** is available for download. Use the download buttons above to get the file."
       
        # Check in root directory
        root_path = os.path.join(os.path.dirname(self.session_cache_path), filename)
        if os.path.exists(root_path):
            return f" **{filename}** is available for download. Use the download buttons above to get the file."
       
        return f" {filename} not found"
   
   
    def get_csv_file_path(self, filename):
        """Get CSV file path for download"""
        # Check in session cache first
        cache_path = os.path.join(self.session_cache_path, filename)
        if os.path.exists(cache_path):
            return os.path.abspath(cache_path)
       
        # Check in root directory
        root_path = os.path.join(os.path.dirname(self.session_cache_path), filename)
        if os.path.exists(root_path):
            return os.path.abspath(root_path)
       
        return None
   
    def get_csv_download_path(self):
        """Get CSV download path for Gradio DownloadButton"""
        # Try sample_data.csv first
        sample_path = self.get_csv_file_path("sample_data.csv")
        if sample_path:
            return sample_path
       
        # Try test_dataset.csv
        test_path = self.get_csv_file_path("test_dataset.csv")
        if test_path:
            return test_path
           
        # Try insurance.csv
        insurance_path = self.get_csv_file_path("insurance.csv")
        if insurance_path:
            return insurance_path
           
        return None
   
    def get_download_path(self):
        """Get CSV file for download"""
        # Only handle CSV files now
        csv_path = self.get_csv_download_path()
        if csv_path:
            return csv_path
           
        return None
   
    def download_file(self):
        """Download file function for Gradio DownloadButton"""
        file_path = self.get_download_path()
        if file_path and os.path.exists(file_path):
            return file_path
        return None
   
    def show_csv_download_button(self):
        """Show CSV download button when files are available"""
        csv_path = self.get_csv_download_path()
        if csv_path:
            return [gr.DownloadButton("Download CSV", value=csv_path, visible=True)]
        else:
            return [gr.DownloadButton("Download CSV", visible=False)]
   
   
   
 
    def save_dialogue(self, chat_history):
        self.conv.save_conv()
        with open(os.path.join(self.session_cache_path, 'system_dialogue.json'), 'w') as f:
            json.dump(chat_history, f, indent=4)
        print(f"Dialogue saved in {os.path.join(self.session_cache_path, 'system_dialogue.json')}.")
 
    def load_dialogue(self, dialogue_path):
        try:
            system_dialogue_path = os.path.join(dialogue_path, 'system_dialogue.json')
            system_config_path = os.path.join(dialogue_path, 'config.json')
            with open(system_dialogue_path, 'r') as f:
                chat_history = json.load(f)
            with open(system_config_path, 'r') as f:
                sys_config = json.load(f)
            self.session_cache_path = sys_config["session_cache_path"]
            self.config["session_cache_path"] = self.session_cache_path
            self.config["chat_history_display"] = chat_history
            self.config["figure_list"] = sys_config["figure_list"]
            return chat_history
        except Exception as e:
            print(f"Failed to load the chat history: {e}")
            return []
 
    def clear_all(self, message, chat_history):
        self.conv.clear()
        return "", []
 
    def update_config(self, conv_model, programmer_model, inspector_model, api_key,
                      base_url_conv_model, base_url_programmer, base_url_inspector,
                      max_attempts, max_exe_time,
                      load_chat, chat_history_path):
 
        self.conv.update_config(conv_model=conv_model, programmer_model=programmer_model, inspector_model=inspector_model, api_key=api_key,
                      base_url_conv_model=base_url_conv_model, base_url_programmer=base_url_programmer, base_url_inspector=base_url_installer,
                      max_attempts=max_attempts, max_exe_time=max_exe_time)
 
        if load_chat == True:
            self.config['chat_history_path'] = chat_history_path
            chat_history = self.load_dialogue(chat_history_path)
            self.config['load_chat'] = load_chat
            return ["### Config Updated!", chat_history]
 
        return "### Config Updated!", []
       
    def debug_system_message(self):
 
        """Debug function to check system message"""
   
        print("\n" + "=" * 80)
   
        print("DEBUG: SYSTEM MESSAGE CHECK")
   
        print("=" * 80)
   
        if len(self.conv.programmer.messages) > 0:
       
            sys_msg = self.conv.programmer.messages[0]
   
            print(f"Role: {sys_msg.get('role')}")
   
            print(f"Content length: {len(sys_msg.get('content', ''))}")
   
            print(f"Content preview (first 500 chars):")
   
            print(sys_msg.get('content', '')[:500])
   
            print(f"\n...Content preview (last 500 chars):")
   
            print(sys_msg.get('content', '')[-500:])
   
        else:
       
            print("ERROR: No system message found!")
   
        print("=" * 80 + "\n")

    def cleanup_old_sessions(self, max_age_hours=12):
        """
        Delete session cache folders older than max_age_hours.
        Called on startup to prevent disk space accumulation.
        """
        import time
        import shutil
        
        print(f" [CLEANUP] Checking for sessions older than {max_age_hours} hours...")
        
        try:
            project_cache_path = to_absolute_path(self.config["project_cache_path"])
            if not os.path.exists(project_cache_path):
                return
            
            current_time = time.time()
            max_age_seconds = max_age_hours * 60 * 60
            deleted_count = 0
            
            for folder_name in os.listdir(project_cache_path):
                folder_path = os.path.join(project_cache_path, folder_name)
                
                # Skip if not a directory
                if not os.path.isdir(folder_path):
                    continue
                
                # Skip current session
                if folder_path == self.session_cache_path:
                    continue
                
                # Check folder age
                folder_mtime = os.path.getmtime(folder_path)
                age_seconds = current_time - folder_mtime
                
                if age_seconds > max_age_seconds:
                    try:
                        shutil.rmtree(folder_path)
                        deleted_count += 1
                        print(f" [CLEANUP] Deleted old session: {folder_name}")
                    except Exception as e:
                        print(f" [CLEANUP] Failed to delete {folder_name}: {e}")
            
            print(f" [CLEANUP] Completed. Deleted {deleted_count} old sessions.")
            
        except Exception as e:
            print(f" [CLEANUP] Error during cleanup: {e}")

    def start_cleanup_thread(self):
        """Start background thread for periodic cleanup"""
        
        def cleanup_loop():
            while True:
                try:
                    # Sleep for 30 minutes between checks
                    time.sleep(30 * 60)
                    
                    # Check kernel idle timeout
                    if hasattr(self.conv, 'kernel'):
                        self.conv.kernel.check_idle_timeout()
                    
                    # Cleanup old sessions every 6 hours
                    self.cleanup_old_sessions(max_age_hours=12)
                    
                except Exception as e:
                    print(f" [CLEANUP THREAD] Error: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        cleanup_thread.start()
        print(" [CLEANUP] Background cleanup thread started")