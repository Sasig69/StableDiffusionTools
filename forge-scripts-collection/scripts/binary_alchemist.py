import modules.scripts as scripts
import gradio as gr
from modules import processing, shared, devices
import torch
import numpy as np
import os
import math

class BinaryAlchemist(scripts.Script):
    def title(self):
        return "The Binary Alchemist (File-to-Latent)"

    def show(self, is_img2img):
        return True

    def ui(self, is_img2img):
        gr.Markdown("### 💾 The Binary Alchemist")
        gr.Markdown("Turn **ANY** file (EXE, DLL, TXT, MP3) into the DNA of your image.")
        
        with gr.Row():
            file_input = gr.File(label="Upload 'Seed' File", file_types=None)
            
        with gr.Row():
            interpret_mode = gr.Dropdown(
                label="Interpretation Mode",
                choices=["Raw Bytes (Digital Grit)", "Bitwise Flow (Liquid Data)", "Frequency Map (Spectral)"],
                value="Raw Bytes (Digital Grit)"
            )
            
            strength = gr.Slider(label="Injection Strength", minimum=0.0, maximum=1.0, value=1.0, info="1.0 = Pure Data. Lower = Blend with Random.")
            
            repeats = gr.Slider(label="Data Density", minimum=1, maximum=10, value=1, step=1, info="Repeat the file data to fill the canvas.")

        return [file_input, interpret_mode, strength, repeats]

    def file_to_tensor(self, file_obj, target_h, target_w, mode, density):
        # 1. Read Raw Bytes
        if file_obj is None: return None
        
        # Gradio File object is a NamedString or temp file path
        # We need to open it as binary
        try:
            with open(file_obj.name, 'rb') as f:
                raw_data = f.read()
        except Exception:
            return None

        # Convert bytes to numpy array of integers (0-255)
        # uint8 is standard byte format
        byte_array = np.frombuffer(raw_data, dtype=np.uint8).astype(np.float32)
        
        if len(byte_array) == 0: return None
        
        # 2. Reshape / Tile to fit Latent Space
        # Target size: 4 Channels * Height * Width
        total_pixels = 4 * target_h * target_w
        
        # If file is too small, tile it
        if len(byte_array) < total_pixels:
            tile_count = math.ceil(total_pixels / len(byte_array))
            byte_array = np.tile(byte_array, tile_count)
            
        # If density > 1, we tile it even more to create "noise" texture
        if density > 1:
            # Take a chunk and repeat it
            chunk_size = len(byte_array) // density
            chunk = byte_array[:chunk_size]
            tile_count = math.ceil(total_pixels / len(chunk))
            byte_array = np.tile(chunk, tile_count)

        # Crop to exact size
        byte_array = byte_array[:total_pixels]
        
        # 3. Mode Processing
        if mode == "Raw Bytes (Digital Grit)":
            # Map 0-255 directly to Latent Intensity
            # Standardize: (X - 127.5) / 127.5 -> range -1 to 1
            tensor_data = (byte_array - 127.5) / 127.5
            
        elif mode == "Bitwise Flow (Liquid Data)":
            # Interpret bytes as angles? No, let's use sin/cos mapping
            # This makes the data smoother
            tensor_data = np.sin(byte_array)
            
        elif mode == "Frequency Map (Spectral)":
            # Treat file as a signal and take FFT
            # Reshape to 2D first
            temp_2d = byte_array.reshape(4, target_h, target_w)
            tensor_data = np.fft.fft2(temp_2d).real
            
        # 4. Final Reshape
        final_tensor = tensor_data.reshape(4, target_h, target_w)
        
        return final_tensor

    def run(self, p, file_input, interpret_mode, strength, repeats):
        
        if file_input is None:
            print("BinaryAlchemist: No file uploaded.")
            return processing.process_images(p)
            
        # 1. Setup
        batch_size = p.batch_size
        c, h, w = 4, p.height // 8, p.width // 8
        device = devices.device
        
        print(f"BinaryAlchemist: Extracting soul of {os.path.basename(file_input.name)}...")

        # 2. Convert File to Latent
        data_map = self.file_to_tensor(file_input, h, w, interpret_mode, repeats)
        
        if data_map is None: return processing.process_images(p)
        
        # 3. Normalize
        # Stable Diffusion latents need Mean~0 and Std~1
        data_tensor = torch.from_numpy(data_map).to(device)
        data_tensor = (data_tensor - data_tensor.mean()) / (data_tensor.std() + 1e-5)
        
        # Expand to batch size
        data_tensor = data_tensor.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        
        # 4. Blend
        if p.seed == -1: p.seed = int(torch.randint(0, 4294967294, (1,)).item())
        torch.manual_seed(p.seed)
        standard_noise = torch.randn(batch_size, c, h, w, device=device)
        
        final_noise = (data_tensor * strength) + (standard_noise * (1.0 - strength))
        final_noise = final_noise / final_noise.std()
        
        # 5. Inject
        p.init_noise = final_noise
        
        return processing.process_images(p)