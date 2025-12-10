import modules.scripts as scripts
import gradio as gr
from modules import processing, shared, devices
import torch
import torch.nn.functional as F
import numpy as np
import random

class LatentBiologist(scripts.Script):
    def title(self):
        return "The Latent Biologist (Turing Morphogenesis)"

    def show(self, is_img2img):
        return True

    def ui(self, is_img2img):
        gr.Markdown("### 🧫 The Latent Biologist")
        gr.Markdown("Grow **Artificial Life** (Turing Patterns) inside the latent space.")
        
        with gr.Row():
            preset = gr.Dropdown(
                label="Biological Species",
                choices=["Coral (Mitosis)", "Zebra (Stripes)", "Bacteria (Spots)", "Fingerprint (Labyrinth)", "Worms (Chaos)"],
                value="Coral (Mitosis)"
            )
            steps = gr.Slider(label="Growth Time (Steps)", minimum=100, maximum=5000, value=1000, step=100)
            zoom = gr.Slider(label="Microscope Zoom", minimum=1, maximum=10, value=2)
            strength = gr.Slider(label="Biological Dominance", minimum=0.0, maximum=1.0, value=0.7)

        return [preset, steps, zoom, strength]

    def solve_gray_scott(self, h, w, steps, feed, kill, device):
        # Initialize U (1.0) and V (0.0)
        u = torch.ones(1, 1, h, w, device=device)
        v = torch.zeros(1, 1, h, w, device=device)
        
        # Seed invader
        seed_noise = torch.rand(1, 1, h, w, device=device)
        v = torch.where(seed_noise > 0.95, torch.tensor(1.0, device=device), v)
        
        # Laplacian Kernel
        kernel = torch.tensor([[0.05, 0.2, 0.05],[0.2, -1.0, 0.2],[0.05, 0.2, 0.05]], device=device).unsqueeze(0).unsqueeze(0)

        du, dv, dt = 0.16, 0.08, 1.0
        
        for _ in range(steps):
            lu = F.conv2d(u, kernel, padding=1)
            lv = F.conv2d(v, kernel, padding=1)
            uvv = u * v * v
            u += (du * lu - uvv + feed * (1 - u)) * dt
            v += (dv * lv + uvv - (feed + kill) * v) * dt
            u = torch.clamp(u, 0, 1)
            v = torch.clamp(v, 0, 1)
            
        return v

    def run(self, p, preset, steps, zoom, strength):
        # 1. Write Metadata (So you know it ran)
        p.extra_generation_params["Biologist Preset"] = preset
        p.extra_generation_params["Biologist Strength"] = strength

        # 2. Setup Magic Numbers
        if preset == "Coral (Mitosis)": feed, kill = 0.0545, 0.062
        elif preset == "Zebra (Stripes)": feed, kill = 0.022, 0.051
        elif preset == "Bacteria (Spots)": feed, kill = 0.035, 0.065
        elif preset == "Fingerprint (Labyrinth)": feed, kill = 0.037, 0.06
        elif preset == "Worms (Chaos)": feed, kill = 0.078, 0.061
        else: feed, kill = 0.0545, 0.062

        # 3. Dimensions
        batch_size = p.batch_size
        h = p.height // 8
        w = p.width // 8
        device = devices.device
        
        print(f"LatentBiologist: Growing {preset}...")
        
        # 4. Simulation
        sim_h = int(h * zoom)
        sim_w = int(w * zoom)
        
        # Run Simulation on GPU
        bio_pattern = self.solve_gray_scott(sim_h, sim_w, steps, feed, kill, device)
        
        # Crop
        start_y = (sim_h - h) // 2
        start_x = (sim_w - w) // 2
        bio_map = bio_pattern[:, :, start_y:start_y+h, start_x:start_x+w]
        
        # 5. Map to Tensor (GPU)
        bio_tensor = torch.zeros(batch_size, 4, h, w, device=device)
        bio_norm = (bio_map - bio_map.mean()) / (bio_map.std() + 1e-5)
        
        for i in range(batch_size):
            bio_tensor[i, 0] = bio_norm * 1.0 
            bio_tensor[i, 1] = bio_norm * 0.5 
            bio_tensor[i, 2] = bio_norm * -0.5
            bio_tensor[i, 3] = bio_norm * 1.0

        # 6. Blend with Standard Noise
        if p.seed == -1: p.seed = int(torch.randint(0, 4294967294, (1,)).item())
        torch.manual_seed(p.seed)
        random_noise = torch.randn(batch_size, 4, h, w, device=device)
        
        final_noise = (bio_tensor * strength) + (random_noise * (1.0 - strength))
        final_noise = final_noise / final_noise.std()
        
        # 7. INJECT (Same method as Frequency Splitter)
        p.init_noise = final_noise
        
        return processing.process_images(p)