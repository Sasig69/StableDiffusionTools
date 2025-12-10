import modules.scripts as scripts
import gradio as gr
from modules import processing, shared, devices
import torch
import random

class TensorBender(scripts.Script):
    def title(self):
        return "The Tensor Bender (Latent Glitch Art)"

    def show(self, is_img2img):
        return True

    def ui(self, is_img2img):
        gr.Markdown("### 🧬 The Tensor Bender")
        gr.Markdown("Directly hacks the 4-channel latent space to create native AI glitches.")
        
        with gr.Row():
            mode = gr.Dropdown(
                label="Glitch Mode", 
                choices=[
                    "Channel Swap (Alien Colors)", 
                    "Dimensional Roll (VHS Tear)", 
                    "Latent Shredder (Strips)", 
                    "Bitcrush (Quantize)", 
                    "Mirror Dimension"
                ], 
                value="Channel Swap (Alien Colors)"
            )
            
            intensity = gr.Slider(label="Glitch Intensity", minimum=0.1, maximum=10.0, value=1.0, step=0.1)
            
            target_channel = gr.Dropdown(
                label="Target Channel", 
                choices=["All", "0 (Structure)", "1 (Green/Mag)", "2 (Blue/Yel)", "3 (Texture)"], 
                value="All",
                visible=False 
            )

        return [mode, intensity, target_channel]

    def run(self, p, mode, intensity, target_channel):
        
        # --- NEW: WRITE METADATA ---
        p.extra_generation_params["TensorBender Mode"] = mode
        p.extra_generation_params["TensorBender Intensity"] = intensity
        # ---------------------------
        
        # 1. Setup Noise Dimensions
        batch_size = p.batch_size
        c, h, w = 4, p.height // 8, p.width // 8
        
        # 2. Generate Base Noise
        if p.seed == -1: p.seed = int(random.randrange(4294967294))
        torch.manual_seed(p.seed)
        noise = torch.randn(batch_size, c, h, w, device=devices.device)
        
        print(f"TensorBender: Applying {mode} with intensity {intensity}...")

        # 3. APPLY THE HACK
        if mode == "Channel Swap (Alien Colors)":
            # Swap Structure (0) with Blue/Yellow (2)
            original = noise.clone()
            noise[:, 0] = original[:, 2] * intensity 
            noise[:, 2] = original[:, 0]             
            
        elif mode == "Dimensional Roll (VHS Tear)":
            # Shift value
            shift = int(10 * intensity)
            
            # Roll entire tensor on Height (Dim 2)
            noise = torch.roll(noise, shifts=shift, dims=2)
            
            # [FIXED] Roll Channel 1 on Width 
            # noise[:, 1] is (Batch, H, W), so Width is Dim 2 (not 3)
            noise[:, 1] = torch.roll(noise[:, 1], shifts=-shift, dims=2)

        elif mode == "Latent Shredder (Strips)":
            freq = int(20 / intensity) if intensity > 0 else 20
            # Create vertical striping mask
            # We iterate over W (dim 3)
            for i in range(w):
                if i % freq == 0:
                    noise[:, :, :, i] *= 0 
            
        elif mode == "Bitcrush (Quantize)":
            step_size = 0.5 * intensity
            noise = torch.round(noise / step_size) * step_size

        elif mode == "Mirror Dimension":
            # Flip entire tensor on Width (Dim 3)
            noise = torch.flip(noise, dims=[3]) 
            
            # [FIXED] Flip Channel 1 back on Width
            # noise[:, 1] is (Batch, H, W), so Width is Dim 2
            noise[:, 1] = torch.flip(noise[:, 1], dims=[2])

        # 4. Inject Hacked Noise
        p.init_noise = noise
        
        return processing.process_images(p)