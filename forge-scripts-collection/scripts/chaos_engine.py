import modules.scripts as scripts
import gradio as gr
from modules import processing, shared, devices
import torch
import numpy as np
import math
from scipy.ndimage import gaussian_filter

class ChaosEngine(scripts.Script):
    def title(self):
        return "The Chaos Engine (Strange Attractors)"

    def show(self, is_img2img):
        return True

    def ui(self, is_img2img):
        gr.Markdown("### 🌪️ The Chaos Engine")
        with gr.Row():
            attractor = gr.Dropdown(
                label="Attractor Type", 
                choices=["Lorenz (Fluid/Gas)", "Clifford (Fractal/Terrain)", "Thomas (Cyclic/Orbital)", "Rossler (Spiral/Deep)"], 
                value="Lorenz (Fluid/Gas)"
            )
            scale = gr.Slider(label="Chaos Zoom", minimum=0.01, maximum=5.0, value=1.0, step=0.01)
            complexity = gr.Slider(label="Time Steps", minimum=1000, maximum=100000, value=20000, step=1000)
            strength = gr.Slider(label="Chaos Strength", minimum=0.0, maximum=1.0, value=0.6)
        return [attractor, scale, complexity, strength]

    # --- 1. CHAOS MATH (Standard) ---
    def get_lorenz(self, n, sigma=10, beta=8/3, rho=28):
        x, y, z = 0.1, 0.0, 0.0; dt = 0.01; points = []
        for _ in range(n):
            dx = sigma * (y - x); dy = x * (rho - z) - y; dz = x * y - beta * z
            x += dx * dt; y += dy * dt; z += dz * dt
            points.append([x, y, z])
        return np.array(points)

    def get_thomas(self, n, b=0.208186):
        x, y, z = 1.1, 1.1, 1.1; dt = 0.05; points = []
        for _ in range(n):
            dx = -b * x + math.sin(y); dy = -b * y + math.sin(z); dz = -b * z + math.sin(x)
            x += dx * dt; y += dy * dt; z += dz * dt
            points.append([x, y, z])
        return np.array(points)

    def get_rossler(self, n, a=0.2, b=0.2, c=5.7):
        x, y, z = 1.0, 1.0, 1.0; dt = 0.01; points = []
        for _ in range(n):
            dx = -y - z; dy = x + a * y; dz = b + z * (x - c)
            x += dx * dt; y += dy * dt; z += dz * dt
            points.append([x, y, z])
        return np.array(points)
        
    def get_clifford(self, w, h, a=1.5, b=-1.8, c=1.6, d=0.9):
        Y, X = np.meshgrid(np.linspace(-2, 2, h), np.linspace(-2, 2, w))
        xn, yn = X, Y
        for _ in range(5):
            xn_new = np.sin(a * yn) + c * np.cos(a * xn); yn_new = np.sin(b * xn) + d * np.cos(b * yn)
            xn, yn = xn_new, yn_new
        return np.stack([xn, yn], axis=0)

    def map_chaos_to_latent(self, points, width, height, scale):
        # Create blank latent
        latent = np.zeros((4, height, width), dtype=np.float32)
        pts = points * scale
        cx, cy = width // 2, height // 2
        xs, ys = pts[:, 0], pts[:, 1]
        
        # Determine Z (Intensity)
        if pts.shape[1] > 2:
            zs = pts[:, 2]
        else:
            zs = np.sin(xs * ys) # Fallback for 2D

        # Map points to grid (Simple Splatting)
        # We iterate only if it's a trajectory (Lorenz/Thomas)
        if len(points.shape) == 2 and points.shape[1] == 3:
            for i in range(len(points)):
                px = int(xs[i] * 4 + cx)
                py = int(ys[i] * 4 + cy)
                
                if 0 <= px < width and 0 <= py < height:
                    val = zs[i]
                    latent[0, py, px] += val       # Structure
                    latent[1, py, px] += val * 0.5 # Color
                    latent[2, py, px] -= val * 0.5 # Color
                    latent[3, py, px] += 1.0       # Texture
        
        # Smooth it out so it's not just dots
        for c in range(4):
            latent[c] = gaussian_filter(latent[c], sigma=2.0)
            
        return latent

    # --- 2. RUN FUNCTION (Simplified to match Frequency Splitter) ---
    def run(self, p, attractor, scale, complexity, strength):
        
        # Metadata
        p.extra_generation_params["Chaos Attractor"] = attractor
        p.extra_generation_params["Chaos Strength"] = strength

        # Dimensions
        batch_size = p.batch_size
        h = p.height // 8
        w = p.width // 8
        device = devices.device
        
        print(f"ChaosEngine: Generating {attractor}...")
        
        # Generate the Chaos Map (CPU Numpy)
        if "Lorenz" in attractor: 
            points = self.get_lorenz(complexity)
            chaos_map = self.map_chaos_to_latent(points, w, h, scale)
        elif "Thomas" in attractor: 
            points = self.get_thomas(complexity)
            chaos_map = self.map_chaos_to_latent(points, w, h, scale)
        elif "Rossler" in attractor: 
            points = self.get_rossler(complexity)
            chaos_map = self.map_chaos_to_latent(points, w, h, scale)
        elif "Clifford" in attractor: 
            # Clifford returns dense field directly
            field = self.get_clifford(w, h)
            chaos_map = np.zeros((4, h, w), dtype=np.float32)
            chaos_map[0] = field[0]
            chaos_map[1] = field[1]
            chaos_map[2] = field[0]*field[1]
            chaos_map[3] = np.sin(field[0]*10)

        # Convert to Tensor on GPU (The Critical Step)
        chaos_tensor = torch.from_numpy(chaos_map).to(device)
        
        # Normalize
        chaos_tensor = (chaos_tensor - chaos_tensor.mean()) / (chaos_tensor.std() + 1e-5)
        
        # Expand to Batch Size
        chaos_tensor = chaos_tensor.unsqueeze(0).repeat(batch_size, 1, 1, 1)

        # Generate Standard Noise (GPU)
        if p.seed == -1: p.seed = int(torch.randint(0, 4294967294, (1,)).item())
        torch.manual_seed(p.seed)
        standard_noise = torch.randn(batch_size, 4, h, w, device=device)
        
        # Blend
        final_noise = (chaos_tensor * strength) + (standard_noise * (1.0 - strength))
        final_noise = final_noise / final_noise.std()

        # INJECT (Directly, just like Freq Splitter)
        p.init_noise = final_noise
        
        return processing.process_images(p)