import gradio as gr
import sys
import json
import os
import re
from modules import script_callbacks

# --- Engine Import ---
try:
    from gallery_dl import extractor, config
    GALLERY_DL_AVAILABLE = True
except ImportError:
    GALLERY_DL_AVAILABLE = False

# --- Config Handling ---
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.json")

def load_config():
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, 'r') as f:
                return json.load(f)
        except:
            pass
    return {}

def save_config(dan_user, dan_key, gel_full_string):
    try:
        data = load_config()
        data.update({
            "danbooru_username": dan_user,
            "danbooru_api_key": dan_key,
            "gelbooru_full_string": gel_full_string
        })
        with open(CONFIG_PATH, 'w') as f:
            json.dump(data, f)
        return "✅ Credentials Saved Successfully!"
    except Exception as e:
        return f"Error saving: {e}"

# --- Logic ---
def get_tags_strict(url, inc_artist, inc_char, inc_copy, rating, dan_user, dan_key, gel_full_string):
    if not GALLERY_DL_AVAILABLE:
        return "❌ Error: 'gallery-dl' library is not installed. Please restart Forge."
    
    if not url:
        return "Please paste a URL first."

    # 1. STRICT DOMAIN CHECK
    valid_domains = ["danbooru", "gelbooru", "safebooru"]
    if not any(domain in url.lower() for domain in valid_domains):
        return "⚠️ Unsupported Site.\nThis tool is strictly configured for Danbooru, Safebooru & Gelbooru."

    # --- 2. INJECT AUTHENTICATION ---
    
    # A. Danbooru / Safebooru
    if ("danbooru" in url or "safebooru" in url) and dan_user and dan_key:
        config.set(("extractor", "danbooru"), "username", dan_user)
        config.set(("extractor", "danbooru"), "password", dan_key)
        print("BooruGrabber: Applied Danbooru Credentials.")

    # B. Gelbooru (Auto-Parse from String)
    if "gelbooru" in url and gel_full_string:
        # Extract User ID
        uid_match = re.search(r'user_id=([0-9]+)', gel_full_string)
        # Extract API Key (alphanumeric long string)
        key_match = re.search(r'api_key=([a-zA-Z0-9]+)', gel_full_string)
        
        if uid_match and key_match:
            gel_id = uid_match.group(1)
            gel_key = key_match.group(1)
            
            config.set(("extractor", "gelbooru"), "user-id", gel_id)
            config.set(("extractor", "gelbooru"), "api-key", gel_key)
            print(f"BooruGrabber: Applied Gelbooru Credentials (ID: {gel_id}).")
        else:
            print("BooruGrabber: Could not parse Gelbooru string. Check format.")

    print(f"BooruGrabber: Fetching {url}...")

    try:
        # 3. Fetch Data
        gen = extractor.find(url)
        data = None
        
        for extr in gen:
            for image_data in extr:
                if isinstance(image_data, dict):
                    data = image_data
                    break 
            if data: break
        
        if not data:
            return "⚠️ No data found. Check your API keys or link."

        # 4. Extract Tags
        final_tags = []
        
        def add(key):
            if key in data and data[key]:
                val = data[key]
                if isinstance(val, list): final_tags.extend(val)
                elif isinstance(val, str): final_tags.append(val)

        if inc_artist: add('tags_artist')
        if inc_char:   add('tags_character')
        if inc_copy:   add('tags_copyright')
        
        if 'tags_general' in data: add('tags_general')
        if not final_tags and 'tags' in data: add('tags')

        # 5. Rating
        if rating and 'rating' in data:
            r = data['rating']
            rating_map = {'s': 'safe', 'g': 'general', 'q': 'questionable', 'e': 'explicit'}
            rat_val = rating_map.get(r, r)
            final_tags.append(f"rating:{rat_val}")

        # 6. Format
        seen = set()
        deduped = [x for x in final_tags if not (x in seen or seen.add(x))]
        formatted = ", ".join(deduped).replace('_', ' ')
        return formatted

    except Exception as e:
        return f"System Error: {str(e)}"

# --- UI Layout ---
def on_ui_tabs():
    cfg = load_config()
    
    with gr.Blocks(analytics_enabled=False) as booru_tab:
        
        with gr.Row():
            # LEFT COLUMN: Controls
            with gr.Column(scale=4):
                gr.Markdown("### 🏷️ Booru Tag Grabber")
                gr.Markdown("Supports: **Danbooru**, **Safebooru** & **Gelbooru**")
                
                with gr.Group():
                    url_input = gr.Textbox(show_label=False, placeholder="Paste Link Here...", elem_id="booru_url_input")
                    fetch_btn = gr.Button("✨ Grab Tags", variant="primary")
                
                with gr.Row(variant="panel"):
                    check_artist = gr.Checkbox(label="Artist", value=True, min_width=80)
                    check_char = gr.Checkbox(label="Character", value=True, min_width=80)
                    check_copy = gr.Checkbox(label="Copyright", value=True, min_width=80)
                    check_rating = gr.Checkbox(label="Rating", value=False, min_width=80)

                gr.HTML("<hr>")
                
                # --- AUTHENTICATION SECTION ---
                with gr.Accordion("🔑 Authentication", open=True):
                    
                    with gr.Tab("Gelbooru"):
                        gr.Markdown("Paste the **Full API String** (e.g. `&api_key=...&user_id=...`)")
                        gel_full_string = gr.Textbox(
                            label="Paste Full String Here", 
                            value=cfg.get("gelbooru_full_string", ""), 
                            placeholder="&api_key=abc...&user_id=123..."
                        )
                        gr.Markdown("[Get Gelbooru Keys](https://gelbooru.com/index.php?page=account&s=options)")
                    
                    with gr.Tab("Danbooru / Safebooru"):
                        dan_user = gr.Textbox(label="Username", value=cfg.get("danbooru_username", ""))
                        dan_key = gr.Textbox(label="API Key", value=cfg.get("danbooru_api_key", ""), type="password")
                        gr.Markdown("[Get Danbooru Keys](https://danbooru.donmai.us/profile)")

                    # --- THE SAVE BUTTON ---
                    save_auth_btn = gr.Button("💾 Save Credentials", variant="secondary")
                    save_msg = gr.Markdown("")

            # RIGHT COLUMN: Results
            with gr.Column(scale=6):
                output_tags = gr.Textbox(label="Extracted Tags", lines=12, show_copy_button=True, interactive=False)
                send_to_txt2img_btn = gr.Button("🚀 Send to txt2img Prompt", variant="secondary")

        # --- Wiring ---
        fetch_btn.click(
            fn=get_tags_strict, 
            inputs=[url_input, check_artist, check_char, check_copy, check_rating, dan_user, dan_key, gel_full_string], 
            outputs=[output_tags]
        )
        
        save_auth_btn.click(
            fn=save_config,
            inputs=[dan_user, dan_key, gel_full_string],
            outputs=[save_msg]
        )
        
        send_to_txt2img_btn.click(fn=None, _js="extract_tags_to_txt2img", inputs=[output_tags], outputs=None)

    return [(booru_tab, "Booru Grabber", "forge_booru_grabber")]

script_callbacks.on_ui_tabs(on_ui_tabs)