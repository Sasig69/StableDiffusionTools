import os
import shutil
import gradio as gr
from modules import scripts, script_callbacks, shared
from modules.images import read_info_from_image
from PIL import Image

GLOBAL_GALLERY_FILES = []

def get_base_dir():
    # Absolute path to your output folder
    path = os.path.abspath(os.path.join(os.getcwd(), "output"))
    if not os.path.exists(path):
        path = os.path.abspath(os.path.join(os.getcwd(), "outputs"))
    return path

def list_subfolders():
    """Finds all subfolders inside 'output' for the dropdown menu."""
    base = get_base_dir()
    subfolders = [base]
    for root, dirs, _ in os.walk(base):
        for d in dirs:
            subfolders.append(os.path.abspath(os.path.join(root, d)))
    return sorted(subfolders)

def update_gallery(folder_path=None, search_query=""):
    global GLOBAL_GALLERY_FILES
    target = folder_path if folder_path else get_base_dir()
    found = []
    if os.path.exists(target):
        # Scan only the selected folder (non-recursive for speed, or remove [0:1] for deep)
        for root, _, files in os.walk(target):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    found.append(os.path.join(root, file))
    
    found.sort(key=os.path.getmtime, reverse=True)
    GLOBAL_GALLERY_FILES = found
    return found

def on_ui_tabs():
    try:
        base_dir = get_base_dir()
        
        with gr.Blocks(analytics_enabled=False) as gallery_ui:
            with gr.Row():
                # LEFT SIDEBAR
                with gr.Column(scale=1, min_width=300):
                    gr.Markdown("### 📂 Folder Navigator")
                    # Using Dropdown instead of FileExplorer for 100% stability
                    folder_drop = gr.Dropdown(
                        choices=list_subfolders(), 
                        value=base_dir, 
                        label="Select Folder",
                        interactive=True
                    )
                    refresh_folders = gr.Button("🔄 Refresh Folder List", variant="secondary")
                    
                    search_bar = gr.Textbox(label="🔍 Search in Folder", placeholder="Prompt keywords...")
                    
                    gr.Markdown("### 🛠️ Actions")
                    fav_btn = gr.Button("⭐ Star Selected", variant="primary")
                    delete_btn = gr.Button("🗑️ Delete File", variant="stop")
                    
                    status_msg = gr.Markdown("Ready")

                # CENTER GALLERY
                with gr.Column(scale=4):
                    image_gallery = gr.Gallery(
                        value=update_gallery(base_dir), 
                        columns=4, 
                        height="85vh", 
                        preview=True
                    )

                # RIGHT INFO
                with gr.Column(scale=1):
                    send_btn = gr.Button("🚀 Send to txt2img", variant="primary")
                    path_display = gr.Textbox(label="File Path", interactive=False)
                    info_box = gr.Textbox(label="Parameters", lines=15, interactive=False)

            # --- EVENT LOGIC ---

            # Folder selection
            folder_drop.change(fn=update_gallery, inputs=[folder_drop], outputs=[image_gallery])
            
            refresh_folders.click(fn=lambda: gr.update(choices=list_subfolders()), outputs=[folder_drop])

            # Search logic
            def do_search(q, f):
                all_files = update_gallery(f)
                if not q: return all_files
                filtered = []
                for p in all_files:
                    try:
                        with Image.open(p) as img:
                            if q.lower() in img.info.get("parameters", "").lower():
                                filtered.append(p)
                    except: continue
                return filtered

            search_bar.submit(fn=do_search, inputs=[search_bar, folder_drop], outputs=[image_gallery])

            # Image selection
            def img_select(evt: gr.SelectData):
                global GLOBAL_GALLERY_FILES
                path = GLOBAL_GALLERY_FILES[evt.index]
                with Image.open(path) as img:
                    info, _ = read_info_from_image(img)
                return path, info
            
            image_gallery.select(fn=img_select, outputs=[path_display, info_box])

            # Favorite Logic
            def make_fav(p):
                if not p: return "Select an image!"
                f_dir = os.path.join(get_base_dir(), "favorites")
                os.makedirs(f_dir, exist_ok=True)
                shutil.copy2(p, os.path.join(f_dir, os.path.basename(p)))
                return "⭐ Saved to Favorites!"

            fav_btn.click(fn=make_fav, inputs=[path_display], outputs=[status_msg])

            # Delete Logic
            def do_delete(p, f):
                if p and os.path.exists(p):
                    os.remove(p)
                return update_gallery(f), "File deleted."

            delete_btn.click(fn=do_delete, inputs=[path_display, folder_drop], outputs=[image_gallery, status_msg])

            # Send to txt2img
            send_btn.click(fn=None, inputs=[info_box], outputs=None, _js="neoGallerySendToTxt")

        return [(gallery_ui, "Gallery", "neo_gallery_tab")]
    except Exception as e:
        print(f"Error: {e}")
        return []

script_callbacks.on_ui_tabs(on_ui_tabs)