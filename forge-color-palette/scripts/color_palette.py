import modules.scripts as scripts
import gradio as gr
from modules import processing
from PIL import Image, ImageDraw, ImageFilter
import random
import math
import colorsys

# --- 1. Color Theory Logic ---
def generate_harmony(hex_color, mode):
    hex_color = str(hex_color).lstrip('#')
    r, g, b = tuple(int(hex_code) for hex_code in (int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)))
    h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
    
    def hsv_to_hex(h, s, v):
        r, g, b = colorsys.hsv_to_rgb(h % 1.0, s, v)
        return '#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255))

    c2, c3 = "#000000", "#FFFFFF"

    if mode == "Complementary":
        c2 = hsv_to_hex(h + 0.5, s, v)
        c3 = hsv_to_hex(h + 0.55, s, v * 0.7)
    elif mode == "Triadic":
        c2 = hsv_to_hex(h + 0.333, s, v)
        c3 = hsv_to_hex(h + 0.666, s, v)
    elif mode == "Analogous":
        c2 = hsv_to_hex(h + 0.08, s, v)
        c3 = hsv_to_hex(h - 0.08, s, v)
    elif mode == "Monochromatic":
        c2 = hsv_to_hex(h, s * 0.6, v * 1.2)
        c3 = hsv_to_hex(h, s, v * 0.4)

    return c2, c3

# --- 2. Pattern Algorithms ---
def create_base_image(width, height, c1, s1, c2, s2, c3, s3, pattern):
    random.seed() 
    
    base_img = Image.new("RGBA", (width, height), (0, 0, 0, 255))
    draw_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(draw_layer)

    def hex_to_rgba(hex_code, strength):
        hex_code = str(hex_code).lstrip('#')
        rgb = tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))
        alpha = int(255 * strength)
        return rgb + (alpha,)

    rgba1 = hex_to_rgba(c1, s1) 
    rgba2 = hex_to_rgba(c2, s2) 
    rgba3 = hex_to_rgba(c3, s3) 
    
    colors = [{'color': rgba1, 'str': s1}, {'color': rgba2, 'str': s2}, {'color': rgba3, 'str': s3}]

    if pattern == "Portrait (Bust)":
        draw.rectangle([0, 0, width, height], fill=rgba3)
        draw.ellipse([-width//4, height//2, width*1.25, height*1.5], fill=rgba2)
        head_w, head_h = width // 2.5, height // 2.5
        draw.ellipse([(width//2)-(head_w//2), (height//3)-(head_h//2), (width//2)+(head_w//2), (height//3)+(head_h//2)], fill=rgba1)
    elif pattern == "Full Body Standing":
        draw.rectangle([0, 0, width, height], fill=rgba3)
        draw.rectangle([width//3, height//5, 2*width//3, height], fill=rgba2)
        head_size = width // 6
        draw.ellipse([(width//2)-head_size, (height//5)-head_size, (width//2)+head_size, (height//5)+head_size], fill=rgba1)
    elif pattern == "Close-Up Face":
        draw.rectangle([0, 0, width, height], fill=rgba3)
        draw.ellipse([0, 0, width, height], fill=rgba2)
        face_w, face_h = width * 0.6, height * 0.7
        draw.ellipse([(width//2)-(face_w//2), (height//2)-(face_h//2), (width//2)+(face_w//2), (height//2)+(face_h//2)], fill=rgba1)
    elif pattern == "Couple (Side-by-Side)":
        draw.rectangle([0, 0, width, height], fill=rgba3)
        draw.ellipse([0, height//3, width//2, height*1.2], fill=rgba1)
        draw.ellipse([width//8, height//6, 3*width//8, height//2], fill=rgba1)
        draw.ellipse([width//2, height//3, width, height*1.2], fill=rgba2)
        draw.ellipse([5*width//8, height//6, 7*width//8, height//2], fill=rgba2)
    elif pattern == "Horizontal Gradient":
        draw.rectangle([0, 0, width, height//3], fill=rgba1)
        draw.rectangle([0, height//3, width, 2*height//3], fill=rgba2)
        draw.rectangle([0, 2*height//3, width, height], fill=rgba3)
    elif pattern == "Vertical Gradient":
        draw.rectangle([0, 0, width//3, height], fill=rgba1)
        draw.rectangle([width//3, 0, 2*width//3, height], fill=rgba2)
        draw.rectangle([2*width//3, 0, width, height], fill=rgba3)
    elif pattern == "Diagonal Split":
        draw.polygon([(0,0), (width,0), (0, height)], fill=rgba1)
        draw.polygon([(width,0), (width, height), (0, height)], fill=rgba3)
        draw_layer.paste(rgba2, [0,0,width,height]) 
        draw.polygon([(0,0), (width//2,0), (0, height//2)], fill=rgba1)
        draw.polygon([(width,height), (width//2,height), (width, height//2)], fill=rgba3)
    elif pattern == "Radial Glow":
        draw.ellipse([0, 0, width, height], fill=rgba3)
        draw.ellipse([width//6, height//6, 5*width//6, 5*height//6], fill=rgba2)
        draw.ellipse([width//3, height//3, 2*width//3, 2*height//3], fill=rgba1)
    elif pattern == "Random Clouds":
        for _ in range(25):
            choice = random.choice(colors)
            cx, cy = random.randint(0, width), random.randint(0, height)
            size = random.randint(width//4, width//2)
            draw.ellipse([cx-size, cy-size, cx+size, cy+size], fill=choice['color'])
    elif pattern == "Chaos Blobs":
        for _ in range(60):
            choice = random.choice(colors)
            cx, cy = random.randint(0, width), random.randint(0, height)
            size = random.randint(width//15, width//8)
            draw.ellipse([cx-size, cy-size, cx+size, cy+size], fill=choice['color'])
    elif pattern == "Spiral Galaxy":
        cx, cy = width//2, height//2
        points = 40
        for i in range(points):
            angle = i * (math.pi / 4)
            dist = i * (min(width, height) // (points * 0.4))
            x, y = cx + math.cos(angle) * dist, cy + math.sin(angle) * dist
            size = width // 6
            draw.ellipse([x-size, y-size, x+size, y+size], fill=colors[i % 3]['color'])
    elif pattern == "Horizontal Waves":
        amplitude, freq = height // 6, 3
        for x in range(0, width, width//20):
            y_base = height // 2
            y_offset = int(math.sin(x / width * math.pi * freq) * amplitude)
            draw.ellipse([x, y_base+y_offset-100, x+40, y_base+y_offset+100], fill=rgba2)
            draw.ellipse([x, y_base+y_offset-200, x+40, y_base+y_offset-50], fill=rgba1)
            draw.ellipse([x, y_base+y_offset+50, x+40, y_base+y_offset+200], fill=rgba3)

    base_img = Image.alpha_composite(base_img, draw_layer)
    final_base = base_img.convert("RGB")
    final_base = final_base.filter(ImageFilter.GaussianBlur(radius=width//25))
    return final_base

# --- 3. The Main Script ---
class ColorPaletteFinalUI(scripts.Script):
    def title(self):
        return "Color Palette (Pro UI)"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):
        gr.Markdown("### 🎨 Smart Composition & Color")

        # --- MAIN SPLIT LAYOUT (Sidebar vs Preview) ---
        with gr.Row():
            
            # === LEFT COLUMN: CONTROLS ===
            with gr.Column(scale=4):
                
                # SECTION 1: SMART HARMONY
                with gr.Row(variant="panel"):
                    with gr.Column(min_width=150):
                        harmony_mode = gr.Dropdown(
                            choices=["Complementary", "Triadic", "Analogous", "Monochromatic"], 
                            value="Complementary", 
                            label="Smart Harmony",
                            interactive=True
                        )
                    with gr.Column(min_width=100):
                        # Empty label for spacing alignment
                        apply_harmony_btn = gr.Button("🎲 Auto-Set Colors", variant="secondary")

                # SECTION 2: MANUAL COLORS
                # We use specific Row variants to keep them tight
                with gr.Group():
                    with gr.Row(variant="compact"):
                        c1 = gr.ColorPicker(show_label=False, value="#FF5733", container=False) 
                        s1 = gr.Slider(label="Focus Opacity", minimum=0.1, maximum=1.0, value=1.0)
                    
                    with gr.Row(variant="compact"):
                        c2 = gr.ColorPicker(show_label=False, value="#33FF57", container=False)
                        s2 = gr.Slider(label="Support Opacity", minimum=0.1, maximum=1.0, value=0.8)
                    
                    with gr.Row(variant="compact"):
                        c3 = gr.ColorPicker(show_label=False, value="#3357FF", container=False)
                        s3 = gr.Slider(label="Back Opacity", minimum=0.1, maximum=1.0, value=1.0)

                # SECTION 3: COMPOSITION
                with gr.Row(variant="panel"):
                    pattern = gr.Dropdown(
                        choices=[
                            "Portrait (Bust)", "Full Body Standing", "Close-Up Face", "Couple (Side-by-Side)",
                            "Horizontal Gradient", "Vertical Gradient", "Diagonal Split", 
                            "Radial Glow", "Random Clouds", "Chaos Blobs", 
                            "Spiral Galaxy", "Horizontal Waves"
                        ], 
                        value="Portrait (Bust)", 
                        label="Layout Pattern"
                    )

                # SECTION 4: BALANCE
                balance = gr.Slider(
                    label="⚖️ Balance (Color vs Prompt)", 
                    minimum=0.0, maximum=1.0, value=0.5, step=0.05
                )

                btn_preview = gr.Button("✨ Update Preview", variant="primary")

            # === RIGHT COLUMN: PREVIEW ===
            with gr.Column(scale=6):
                preview_img = gr.Image(
                    label="Composition Preview", 
                    type="pil", 
                    interactive=False, 
                    height=450,  # Taller height for better visibility
                    show_download_button=False
                )

        # --- EVENTS ---
        apply_harmony_btn.click(fn=generate_harmony, inputs=[c1, harmony_mode], outputs=[c2, c3])

        def on_preview(c1, s1, c2, s2, c3, s3, pat):
            return create_base_image(512, 512, c1, s1, c2, s2, c3, s3, pat)

        btn_preview.click(
            fn=on_preview,
            inputs=[c1, s1, c2, s2, c3, s3, pattern],
            outputs=[preview_img]
        )

        return [c1, s1, c2, s2, c3, s3, pattern, balance]

    def run(self, p, c1, s1, c2, s2, c3, s3, pattern, balance):
        print(f"ColorPalette: Creating {p.width}x{p.height} base...")
        base_img = create_base_image(p.width, p.height, c1, s1, c2, s2, c3, s3, pattern)

        p.init_images = [base_img]
        # Map balance 0.0-1.0 to denoise 0.65-0.95
        p.denoising_strength = 0.65 + (balance * 0.30)
        
        proc = processing.process_images(p)
        return proc