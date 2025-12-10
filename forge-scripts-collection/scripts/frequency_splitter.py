import modules.scripts as scripts
import gradio as gr
from modules import processing, shared, devices
import torch
import numpy as np

class FrequencySplitter(scripts.Script):
    def title(self):
        return "The Frequency Splitter (Seed Mixer)"

    def show(self, is_img2img):
        return True

    def ui(self, is_img2img):
        gr.Markdown("### 🎛️ Frequency Splitter")
        gr.Markdown("Combine the **Structure** of one seed with the **Texture** of another.")
        
        with gr.Row():
            with gr.Column():
                seed_struct = gr.Number(label="Structure Seed (Low Freq)", value=-1, precision=0)
                seed_detail = gr.Number(label="Texture Seed (High Freq)", value=-1, precision=0)
            
            with gr.Column():
                cutoff = gr.Slider(
                    label="Crossover Frequency", 
                    minimum=0, maximum=64, value=4, 
                    info="Lower = More Texture dominance. Higher = More Structure dominance."
                )
                
                softness = gr.Slider(label="Filter Softness", minimum=0, maximum=10, value=2)

        return [seed_struct, seed_detail, cutoff, softness]

    def mix_frequencies(self, noise_low, noise_high, cutoff, softness, device):
        # 1. Convert both to Frequency Domain
        fft_low = torch.fft.fft2(noise_low)
        fft_high = torch.fft.fft2(noise_high)
        
        # 2. Create the Filter Mask
        b, c, h, w = noise_low.shape
        fx = torch.fft.fftfreq(w, device=device)
        fy = torch.fft.fftfreq(h, device=device)
        wx, wy = torch.meshgrid(fx, fy, indexing='xy')
        
        # Calculate distance from center (Frequency Magnitude)
        # Low frequencies are near 0
        freq_rad = torch.sqrt(wx**2 + wy**2) * min(w, h)
        
        # 3. Create Low-Pass and High-Pass filters
        # Sigmoid function for soft blending
        # Mask = 1.0 at Low Freq, 0.0 at High Freq
        mask = 1.0 / (1.0 + torch.exp((freq_rad - cutoff) / (softness + 0.1)))
        
        # Expand mask for channels: (H, W) -> (1, 1, H, W)
        mask = mask.unsqueeze(0).unsqueeze(0)
        
        # 4. Mix
        # Output = (Low * Mask) + (High * (1-Mask))
        fft_mixed = (fft_low * mask) + (fft_high * (1.0 - mask))
        
        # 5. Inverse FFT
        mixed = torch.fft.ifft2(fft_mixed).real
        
        # 6. Normalize (Critical for SD)
        mixed = mixed / mixed.std()
        
        return mixed

    def run(self, p, seed_struct, seed_detail, cutoff, softness):
        
        # 1. Setup Seeds
        if seed_struct == -1: seed_struct = int(torch.randint(0, 4294967294, (1,)).item())
        if seed_detail == -1: seed_detail = int(torch.randint(0, 4294967294, (1,)).item())
        
        # Override main seed to match structure (for metadata consistency)
        p.seed = seed_struct
        
        # 2. Dimensions
        batch_size = p.batch_size
        c = 4
        h = p.height // 8
        w = p.width // 8
        
        print(f"FreqSplitter: Mixing Structure({seed_struct}) + Detail({seed_detail})...")

        # 3. Generate the Two Noises
        torch.manual_seed(seed_struct)
        noise_struct = torch.randn(batch_size, c, h, w, device=devices.device)
        
        torch.manual_seed(seed_detail)
        noise_detail = torch.randn(batch_size, c, h, w, device=devices.device)
        
        # 4. Perform Frequency Mixing
        final_noise = self.mix_frequencies(
            noise_struct, 
            noise_detail, 
            cutoff, 
            softness, 
            devices.device
        )
        
        # 5. Inject
        p.init_noise = final_noise
        
        return processing.process_images(p)