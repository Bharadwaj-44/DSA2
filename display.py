import gradio
import html
import re
import os

def display_text(text):
    return f"""<div style="border: 1px solid #ccc; padding: 10px; margin-bottom: 10px;"><p>{text}</p></div>"""

def display_image(path):
    # Convert to absolute path and forward slashes for web compatibility
    abs_path = os.path.abspath(path)
    web_path = abs_path.replace('\\', '/')
    
    # Try multiple approaches for better Gradio compatibility
    try:
        # Approach 1: Try to make it relative to the cache directory
        if 'cache' in abs_path:
            cache_index = abs_path.find('cache')
            rel_path = abs_path[cache_index:]
            rel_path = rel_path.replace('\\', '/')
        else:
            rel_path = web_path
    except:
        rel_path = web_path
    
    # Try base64 encoding approach for better compatibility
    try:
        import base64
        with open(abs_path, 'rb') as img_file:
            img_data = base64.b64encode(img_file.read()).decode('utf-8')
            img_html = f'<div style="text-align: center; margin: 10px 0; padding: 10px; border: 1px solid #ddd; border-radius: 8px; background-color: #f9f9f9;"><img src="data:image/png;base64,{img_data}" style="max-width: 100%; height: auto; border: 1px solid #ccc; border-radius: 4px;" alt="Generated Chart"></div>'
            return img_html
    except Exception as e:
        print(f"Base64 encoding failed: {e}")
    
    # Fallback to simple text with file info
    text_info = f'📊 **Chart Generated**: {os.path.basename(path)}\n🔗 **File Path**: {rel_path}\n💡 **Note**: Chart saved successfully but may not display in browser due to security restrictions.'
    
    return text_info


def display_exe_results(text):
    escaped_text = html.escape(text)
    return f"""<details style="border: 1px solid #ccc; padding: 10px; margin-bottom: 10px;"><summary style="font-weight: bold; cursor: pointer;">✅Click to view execution results</summary><pre>{escaped_text}</pre></details>"""


def display_download_file(path, filename):
    abs_path = os.path.abspath(path)
    # For Gradio file serving, we need to use the file= prefix with the absolute path
    file_url = f"file={abs_path}"
    return f"""<div style="border: 1px solid #ccc; padding: 10px; margin-bottom: 10px;"><a href="{file_url}" download="{filename}" style="font-weight: bold; color: #007bff;">📥 {filename}</a></div>"""

def display_csv_file(path, filename):
    """Display file as a simple downloadable link"""
    abs_path = os.path.abspath(path)
    if os.path.exists(abs_path):
        # Create a simple download link using Gradio's file serving
        file_url = f"file={abs_path}"
        return f"""<div style="margin: 5px 0;">
            <a href="{file_url}" download="{filename}" 
               style="color: #007bff; text-decoration: underline; font-weight: bold;">
                📥 {filename}
            </a>
        </div>"""
    else:
        return f"""<div style="color: #dc3545; font-size: 12px;">❌ {filename} not found</div>"""

def display_ml_model_file(path, filename):
    """Display ML model file as a simple downloadable link"""
    abs_path = os.path.abspath(path)
    if os.path.exists(abs_path):
        # Create a simple download link using Gradio's file serving
        file_url = f"file={abs_path}"
        return f"""<div style="margin: 5px 0;">
            <a href="{file_url}" download="{filename}" 
               style="color: #007bff; text-decoration: underline; font-weight: bold;">
                📥 {filename}
            </a>
        </div>"""
    else:
        return f"""<div style="color: #dc3545; font-size: 12px;">❌ {filename} not found</div>"""

def get_csv_download_path(path, filename):
    """Get CSV file path for Gradio download button"""
    abs_path = os.path.abspath(path)
    if os.path.exists(abs_path):
        return abs_path
    return None

def suggestion_html(suggestions: list) -> str:
    buttons_html = ""
    for suggestion in suggestions:
        buttons_html += f"""<button class='suggestion-btn'>{suggestion}</button>"""
    return f"<div>{buttons_html}</div>"


def display_suggestions(prog_response, chat_history_display_last):
    '''
        replace：
            Next, you can:
            [1] Do something...
            [2] Do something else...

        by：
            <div>
                <button class="suggestion-btn" data-bound="true">...</button>
                <button class="suggestion-btn" data-bound="true">...</button>
            </div>
    '''
    suggest_list = re.findall(r'\[\d+\]\s*(.*)', prog_response)
    if suggest_list:

        button_html = suggestion_html(suggest_list)

        pattern = r'(Next, you can:)(.*?)(?=(?:<br>)?\Z)'
        chat_history_display_last = re.sub(pattern, r'\1\n' + button_html, chat_history_display_last, flags=re.DOTALL)

    return chat_history_display_last
