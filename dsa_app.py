import sys
import asyncio
 
# Fix for Python 3.12 + uvicorn + Gradio compatibility
if sys.platform == 'win32' and sys.version_info >= (3, 12):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
 
import os
import time
print(" Starting DSA Application...")
start_time = time.time()
# Set matplotlib backend before any other imports
os.environ['MPLBACKEND'] = 'Agg'
print(" Loading matplotlib...")
# ... rest continues as before
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

print(" Loading Gradio...")
import gradio as gr
print(" Loading frontend components...")
from front_end.js import js
from front_end.css import css
print("📦 Loading DSA core...")
from DSA import DSA
from utils.utils import to_absolute_path

# Additional matplotlib configuration for Gradio compatibility
print("🔧 Configuring matplotlib...")
import matplotlib.pyplot as plt
plt.ioff()  # Turn off interactive mode

print(f"✅ Libraries loaded in {time.time() - start_time:.2f} seconds")

def launch_app():
    dsa = DSA(config_path='config.yaml')
    with gr.Blocks(theme=gr.themes.Soft(), css=css, js=js) as demo:

        with gr.Tab("Data Science Agent"):
            gr.HTML("<H1>Welcome to Elevance Data Science Agent! Easy Data Analysis!</H1>")
            upload_status = gr.HTML(value="", visible=False)
            chatbot = gr.Chatbot(value=dsa.conv.chat_history_display, height=600, label="Data Science Agent", show_copy_button=True, type="tuples", render_markdown=True)
            with gr.Group():
                with gr.Row(equal_height=True):
                    upload_btn = gr.UploadButton(label="Upload Data", file_types=[".csv", ".xlsx"], scale=1)
                    msg = gr.Textbox(show_label=False, placeholder="Sent message to LLM", scale=6, elem_id="chatbot_input")
                    submit = gr.Button("Submit", scale=1)
            with gr.Row(equal_height=True):
                board = gr.Button(value="Show/Update DataFrame", elem_id="df_btn", elem_classes="df_btn")
                export_notebook = gr.Button(value="Notebook")
                down_notebook = gr.DownloadButton("Download Notebook", visible=False)
                generate_report = gr.Button(value="Generate Report")
                down_report = gr.DownloadButton("Download Report", visible=False)
                download_csv = gr.DownloadButton("Download CSV", visible=False)

                edit = gr.Button(value="Edit Code", elem_id="ed_btn", elem_classes="ed_btn")
                clear = gr.ClearButton(value="Clear All")

            with gr.Group():
                with gr.Row(visible=False, elem_id="ed", elem_classes="ed"):
                    code = gr.Code(label="Code", scale=6)
                    code_btn = gr.Button("Submit Code", scale=1)
            code_btn.click(fn=dsa.chat_streaming, inputs=[msg, chatbot, code], outputs=[msg, chatbot]).then(
                dsa.conv.stream_workflow, inputs=[chatbot, code], outputs=chatbot)

            df = gr.Dataframe(visible=False, elem_id="df", elem_classes="df")

            def clear_upload_status():
                return gr.HTML(visible=False)
            
            upload_btn.upload(fn=clear_upload_status, outputs=upload_status).then(
                fn=dsa.add_file_with_feedback, inputs=upload_btn, outputs=[upload_status]
            )
            msg.submit(dsa.chat_streaming, [msg, chatbot], [msg, chatbot], queue=False).then(
                dsa.conv.stream_workflow, chatbot, chatbot
            )
            submit.click(dsa.chat_streaming, [msg, chatbot], [msg, chatbot], queue=False).then(
                dsa.conv.stream_workflow, chatbot, chatbot
            )
            board.click(dsa.open_board, inputs=[], outputs=df)
            edit.click(dsa.rendering_code, inputs=None, outputs=code)
            export_notebook.click(dsa.export_code, inputs=None, outputs=[export_notebook, down_notebook])
            down_notebook.click(dsa.down_notebook, inputs=None, outputs=[export_notebook, down_notebook])
            generate_report.click(dsa.generate_report, inputs=[chatbot], outputs=[generate_report, down_report])
            down_report.click(dsa.down_report, inputs=None, outputs=[generate_report, down_report])
            download_csv.click(fn=dsa.download_file, outputs=download_csv)
            clear.click(fn=dsa.clear_all, inputs=[msg, chatbot], outputs=[msg, chatbot])

        # The Configuration Page
        with gr.Tab("Configuration"):
            gr.Markdown("# System Configuration for Data Science Agent")
            with gr.Row():
                conv_model = gr.Textbox(value="gpt-3.5-turbo", label="Conversation Model")
                programmer_model = gr.Textbox(value="gpt-3.5-turbo", label="Programmer Model")
                inspector_model = gr.Textbox(value="gpt-3.5-turbo", label="Inspector Model")
            
            api_key = gr.Textbox(label="API Key", type="password", placeholder="Input Your API key")
            with gr.Row():
                base_url_conv_model = gr.Textbox(value='https://api.openai.com/v1', label="Base URL (Conv Model)")
                base_url_programmer = gr.Textbox(value='https://api.openai.com/v1', label="Base URL (Programmer)")
                base_url_inspector = gr.Textbox(value='https://api.openai.com/v1', label="Base URL (Inspector)")

            with gr.Row():
                max_attempts = gr.Number(value=5, label="Max Attempts", precision=0)
                max_exe_time = gr.Number(value=18000, label="Max Execution Time (s)", precision=0)
            with gr.Row():            
                load_chat = gr.Checkbox(value=False, label="Load from Cache")
                chat_history_path = gr.Textbox(label="Chat History Path", visible=False, interactive=True)
                
            save_btn = gr.Button("Save Configuration", variant="primary")
            status_output = gr.Markdown("")
            
            def toggle_chat_history_path(load_chat_checked):
                return gr.Textbox(visible=load_chat_checked, interactive=True)
            
            save_btn.click(
                fn=dsa.update_config,
                inputs=[
                    conv_model, programmer_model, inspector_model, api_key,
                    base_url_conv_model, base_url_programmer, base_url_inspector,
                    max_attempts, max_exe_time,
                    load_chat, chat_history_path
                ],
                outputs=[status_output, chatbot]
            )

            load_chat.change(
                fn=toggle_chat_history_path,
                inputs=load_chat,
                outputs=chat_history_path
            )

    # Get all possible cache paths
    allowed_paths = [
        to_absolute_path(dsa.config["project_cache_path"]), 
        to_absolute_path("cache"),
        to_absolute_path(dsa.session_cache_path),
        dsa.session_cache_path,  # Direct session cache path
        os.path.dirname(dsa.session_cache_path),  # Parent directory
        to_absolute_path(".")  # Project root
    ]
    
    demo.launch(
    server_name="0.0.0.0",      # Listen on all interfaces (for K8s)
    server_port=8000,            # Your exposed port
    allowed_paths=allowed_paths,
    share=False,                 #  NO public sharing!
    inbrowser=False,             # Don't try to open browser
    ssl_verify=False             # For internal corporate certs
    )

if __name__ == '__main__':
    launch_app()
