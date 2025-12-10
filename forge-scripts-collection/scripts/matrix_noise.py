import modules.scripts as scripts
import gradio as gr
from modules import processing, shared, devices
import torch
import numpy as np
import math

class MatrixNoiseInjector(scripts.Script):
    def title(self):
        return "The Matrix Injector (Geometric Noise)"

    def show(self, is_img2img):
        return True

    def ui(self, is_img2img):
        gr.Markdown("### 🧬 Matrix Noise Injector")
        gr.Markdown("Injects mathematical geometry into the initial static.")
        
        with gr.Row():
            pattern_type = gr.Dropdown(
                label="Pattern Type", 
                choices=["Checkerboard", "Radial Rings", "Spiral Galaxy", "Vertical Bars", "Plasma Waves"], 
                value="Checkerboard"
            )
            
            frequency = gr.Slider(label="Pattern Density", minimum=1, maximum=50, value=10, step=1)
            strength = gr.Slider(label="Injection Strength", minimum=0.0, maximum=1.0, value=0.3, info="0.3 is subtle. 0.8 is strict.")

        return [pattern_type, frequency, strength]

    def generate_pattern(self, shape, mode, freq, device):
        b, c, h, w = shape
        # Create coordinate grid
        x = torch.linspace(-1, 1, w, device=device)
        y = torch.linspace(-1, 1, h, device=device)
        gy, gx = torch.meshgrid(y, x, indexing='ij')
        
        pattern = torch.zeros(b, c, h, w, device=device)
        
        if mode == "Checkerboard":
            # Sin(x) * Sin(y) creates a grid
            pattern = torch.sin(gx * freq * math.pi) * torch.sin(gy * freq * math.pi)
            
        elif mode == "Radial Rings":
            # Distance from center
            rad = torch.sqrt(gx**2 + gy**2)
            pattern = torch.sin(rad * freq * math.pi)
            
        elif mode == "Vertical Bars":
            pattern = torch.sin(gx * freq * math.pi)
            
        elif mode == "Spiral Galaxy":
            rad = torch.sqrt(gx**2 + gy**2)
            angle = torch.atan2(gy, gx)
            pattern = torch.sin((rad * freq) + angle * 5)
            
        elif mode == "Plasma Waves":
            pattern = torch.sin(gx * freq) + torch.sin(gy * freq) + torch.sin((gx + gy) * freq * 0.5)

        # Normalize pattern to act like Gaussian Noise (Mean 0, Std 1)
        # This is CRITICAL so the AI doesn't produce "burned" images
        pattern = (pattern - pattern.mean()) / (pattern.std() + 1e-5)
        
        return pattern

    def run(self, p, pattern_type, frequency, strength):
        
        # 1. Setup Dimensions (Latent space is 1/8th of pixel space)
        batch_size = p.batch_size
        channels = 4 # SD Standard
        h = p.height // 8
        w = p.width // 8
        
        print(f"MatrixInjector: Injecting {pattern_type}...")

        # 2. Generate The Pattern
        geo_noise = self.generate_pattern(
            (batch_size, channels, h, w), 
            pattern_type, 
            frequency, 
            devices.device
        )
        
        # 3. Generate Standard Random Noise (The chaos)
        # We handle seeding manually to ensure consistency
        if p.seed == -1: p.seed = int(torch.randint(0, 4294967294, (1,)).item())
        torch.manual_seed(p.seed)
        random_noise = torch.randn(batch_size, channels, h, w, device=devices.device)
        
        # 4. Blend them
        # Weighted average: (Pattern * Strength) + (Random * (1-Strength))
        # Then re-normalize to ensure the math stays valid for diffusion
        final_noise = (geo_noise * strength) + (random_noise * (1.0 - strength))
        final_noise = (final_noise - final_noise.mean()) / final_noise.std()
        
        # 5. Inject
        p.init_noise = final_noise
        
        return processing.process_images(p)