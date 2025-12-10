import modules.scripts as scripts
import gradio as gr
from modules import processing, shared, devices
import torch
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal
import os

class NeuralResonator(scripts.Script):
    def title(self):
        return "The Neural Resonator (Audio-to-Latent)"

    def show(self, is_img2img):
        return True

    def ui(self, is_img2img):
        gr.Markdown("### 🎹 The Neural Resonator")
        gr.Markdown("Injects **Audio Spectrograms** into the diffusion process. Turn sound into texture.")
        
        with gr.Row():
            audio_input = gr.Audio(label="Upload Audio", type="filepath")
            
        with gr.Row():
            mapping_mode = gr.Dropdown(
                label="Mapping Strategy",
                choices=["Direct (Spectrogram)", "Circular (Vinyl Record)", "Rhythmic (Beat Pulse)"],
                value="Direct (Spectrogram)"
            )
            
            strength = gr.Slider(label="Resonance Strength", minimum=0.0, maximum=1.0, value=0.8)
            zoom = gr.Slider(label="Time Zoom", minimum=0.1, maximum=5.0, value=1.0, info="Zoom into a specific split-second of audio.")

        return [audio_input, mapping_mode, strength, zoom]

    def audio_to_spectrogram(self, filepath, target_h, target_w, zoom):
        # 1. Load Audio
        if not filepath: return None
        rate, data = wav.read(filepath)
        
        # Convert to mono if stereo
        if len(data.shape) > 1:
            data = data.mean(axis=1)
            
        # 2. Compute Spectrogram
        # We want high resolution for latents
        f, t, Sxx = scipy.signal.spectrogram(data, fs=rate, nperseg=512)
        
        # Sxx is (Frequency, Time)
        # We need to map this to (Height, Width)
        
        # Logarithmic scale matches human hearing (and looks better)
        Sxx = np.log1p(Sxx)
        
        # 3. Crop/Resize to Target Dimensions
        # Sxx shape might be (257, 10000)
        
        # We select a slice based on 'zoom'
        # Center the crop
        spec_h, spec_w = Sxx.shape
        center_x = spec_w // 2
        crop_w = int(target_w * (1/zoom) * 10) # Arbitrary scaling factor
        
        start_x = max(0, center_x - crop_w // 2)
        end_x = min(spec_w, center_x + crop_w // 2)
        
        # Slice
        Sxx_crop = Sxx[:, start_x:end_x]
        
        # Resize using simple interpolation (using scipy or torch)
        # Let's use Torch for easy resizing
        tensor = torch.from_numpy(Sxx_crop).float().unsqueeze(0).unsqueeze(0) # (1, 1, H, W)
        
        # Resize to (Latent Height, Latent Width)
        # target_h/w are usually 64 or 128
        resized = torch.nn.functional.interpolate(tensor, size=(target_h, target_w), mode='bilinear')
        
        # Squeeze back to (H, W)
        return resized.squeeze().numpy()

    def map_to_channels(self, spectrogram, height, width, mode):
        # Initialize 4 channels
        latent = np.zeros((4, height, width), dtype=np.float32)
        
        if spectrogram is None: return latent

        if mode == "Direct (Spectrogram)":
            # Map Frequencies directly to vertical axis
            # Low Freq (Bottom) -> Structure (Channel 0)
            # Mid Freq -> Color (Channel 1, 2)
            # High Freq (Top) -> Texture (Channel 3)
            
            # We split the spectrogram vertically into bands
            h = spectrogram.shape[0]
            band1 = int(h * 0.25) # Bass
            band2 = int(h * 0.50) # Mids
            band3 = int(h * 0.75) # Highs
            
            # We assume spectrogram represents intensity
            # Normalize first
            spec_norm = (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min() + 1e-5)
            
            # Simple mapping: Use the whole spectrogram intensity for structure
            latent[0] = spec_norm # Structure follows sound intensity
            
            # Invert for colors (Complimentary)
            latent[1] = np.sin(spec_norm * np.pi * 2) 
            latent[2] = np.cos(spec_norm * np.pi * 2)
            
            # High frequency noise for texture
            latent[3] = spec_norm * np.random.randn(height, width)
            
        elif mode == "Circular (Vinyl Record)":
            # Warp the spectrogram into a circle (Polar coordinates)
            # This creates "Tunnel" or "Eye" effects from sound
            Y, X = np.ogrid[:height, :width]
            center = (height//2, width//2)
            dist = np.sqrt((X - center[1])**2 + (Y - center[0])**2)
            angle = np.arctan2(Y - center[0], X - center[1])
            
            # Map Distance to Time, Angle to Frequency
            # This is complex, so we approximate with simple radial waves driven by audio intensity
            avg_intensity = spectrogram.mean()
            latent[0] = np.sin(dist / (avg_intensity * 10 + 1))
            latent[1] = np.cos(angle * 5)
            
        elif mode == "Rhythmic (Beat Pulse)":
            # Takes the rhythmic volume envelope
            volume = spectrogram.mean(axis=0) # 1D array over time
            # Tile it vertically to create "Bars"
            bars = np.tile(volume, (height, 1))
            latent[0] = bars
            latent[1] = bars * -1
            
        return latent

    def run(self, p, audio_input, mapping_mode, strength, zoom):
        
        if not audio_input:
            print("NeuralResonator: No audio found. Skipping.")
            return processing.process_images(p)
            
        # 1. Setup
        batch_size = p.batch_size
        c, h, w = 4, p.height // 8, p.width // 8
        device = devices.device
        
        print(f"NeuralResonator: Listening to audio...")

        # 2. Process Audio
        # Generate a spectrogram matched to the latent dimensions
        spec = self.audio_to_spectrogram(audio_input, h, w, zoom)
        
        # 3. Map to Latent Space
        audio_map = self.map_to_channels(spec, h, w, mapping_mode)
        
        # 4. Normalize
        audio_tensor = torch.from_numpy(audio_map).to(device)
        audio_tensor = (audio_tensor - audio_tensor.mean()) / (audio_tensor.std() + 1e-5)
        audio_tensor = audio_tensor.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        
        # 5. Blend
        if p.seed == -1: p.seed = int(torch.randint(0, 4294967294, (1,)).item())
        torch.manual_seed(p.seed)
        standard_noise = torch.randn(batch_size, c, h, w, device=device)
        
        final_noise = (audio_tensor * strength) + (standard_noise * (1.0 - strength))
        final_noise = final_noise / final_noise.std()
        
        # 6. Inject
        p.init_noise = final_noise
        
        return processing.process_images(p)