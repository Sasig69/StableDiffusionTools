import modules.scripts as scripts
import gradio as gr
from modules import processing, shared, devices
import torch
import numpy as np

class LatentAlchemist(scripts.Script):
    def title(self):
        return "The Latent Alchemist (Structure/Color/Detail Splitter)"

    def show(self, is_img2img):
        return True

    def ui(self, is_img2img):
        gr.Markdown("### ⚗️ The Latent Alchemist")
        gr.Markdown("Independently control the **Pose**, **Colors**, and **Texture** of your generation.")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### 1. The Ingredients")
                seed_struct = gr.Number(label="Structure Seed (Pose/Shape)", value=-1, precision=0)
                seed_color  = gr.Number(label="Color Seed (Palette/Mood)", value=-1, precision=0)
                seed_detail = gr.Number(label="Texture Seed (Grain/Detail)", value=-1, precision=0)
            
            with gr.Column():
                gr.Markdown("#### 2. The Mixing")
                # Frequency Cutoff
                freq_cutoff = gr.Slider(label="Detail Threshold", minimum=0, maximum=32, value=3, info="Lower = Pure Structure. Higher = More Detail from Structure seed.")
                
                # Color Strength
                color_lock = gr.Checkbox(label="Lock Color Channels?", value=True, info="If checked, uses Color Seed exclusively for Channels 1 & 2.")

        return [seed_struct, seed_color, seed_detail, freq_cutoff, color_lock]

    def mix_frequencies(self, noise_low, noise_high, cutoff, device):
        """
        Combines Low Freqs from 'noise_low' and High Freqs from 'noise_high'
        """
        # FFT transform
        fft_low = torch.fft.fft2(noise_low)
        fft_high = torch.fft.fft2(noise_high)
        
        b, c, h, w = noise_low.shape
        
        # Create Frequency Mask
        fx = torch.fft.fftfreq(w, device=device)
        fy = torch.fft.fftfreq(h, device=device)
        wx, wy = torch.meshgrid(fx, fy, indexing='xy')
        freq_rad = torch.sqrt(wx**2 + wy**2) * min(w, h)
        
        # Soft Sigmoid Mask
        mask = 1.0 / (1.0 + torch.exp((freq_rad - cutoff) / 1.5))
        mask = mask.unsqueeze(0).unsqueeze(0)
        
        # Blend
        fft_mixed = (fft_low * mask) + (fft_high * (1.0 - mask))
        
        # Inverse FFT
        mixed = torch.fft.ifft2(fft_mixed).real
        return mixed

    def run(self, p, seed_struct, seed_color, seed_detail, freq_cutoff, color_lock):
        
        # 1. SETUP SEEDS
        # If -1, generate random ones
        if seed_struct == -1: seed_struct = int(torch.randint(0, 4294967294, (1,)).item())
        if seed_color == -1: seed_color = int(torch.randint(0, 4294967294, (1,)).item())
        if seed_detail == -1: seed_detail = int(torch.randint(0, 4294967294, (1,)).item())
        
        # Set main seed to structure for metadata logs
        p.seed = seed_struct
        
        # Dimensions
        batch_size = p.batch_size
        c = 4
        h = p.height // 8
        w = p.width // 8
        device = devices.device
        
        print(f"Alchemist: 🧱{seed_struct} | 🎨{seed_color} | ░{seed_detail}")

        # 2. GENERATE RAW NOISES
        # Structure Noise (Base)
        torch.manual_seed(seed_struct)
        noise_struct = torch.randn(batch_size, c, h, w, device=device)
        
        # Texture Noise (Detail)
        torch.manual_seed(seed_detail)
        noise_detail = torch.randn(batch_size, c, h, w, device=device)
        
        # Color Noise (Palette)
        torch.manual_seed(seed_color)
        noise_color = torch.randn(batch_size, c, h, w, device=device)

        # 3. STEP A: FREQUENCY BLEND (Structure + Detail)
        # We mix the Structure Seed (Low Freq) with Detail Seed (High Freq)
        # This creates a noise with the "Pose" of A and "Grain" of B.
        noise_luminance = self.mix_frequencies(noise_struct, noise_detail, freq_cutoff, device)
        
        # 4. STEP B: CHANNEL BLEND (Luminance + Color)
        # If Color Lock is ON:
        # We take Channels 0 & 3 (Light/Structure) from the Frequency Mix
        # We take Channels 1 & 2 (Color/Tint) from the Color Seed
        
        final_noise = noise_luminance.clone()
        
        if color_lock:
            # Channel 0: Structure/Light (Keep from Freq Mix)
            # Channel 1: Color Green/Magenta (Take from Color Seed)
            final_noise[:, 1] = noise_color[:, 1] 
            # Channel 2: Color Blue/Yellow (Take from Color Seed)
            final_noise[:, 2] = noise_color[:, 2]
            # Channel 3: Texture/Structure (Keep from Freq Mix)
            
            # Note: We don't apply frequency mixing to color channels usually, 
            # because color is low-frequency by nature. Using the raw color seed 
            # ensures the "Vibe" is pure.
            
        # 5. NORMALIZE
        # Essential to prevent "Fried" images
        final_noise = final_noise / final_noise.std()

        # 6. INJECT
        p.init_noise = final_noise
        
        return processing.process_images(p)