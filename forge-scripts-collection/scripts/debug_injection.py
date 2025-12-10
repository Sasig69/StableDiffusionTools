import modules.scripts as scripts
from modules import processing
import torch

class DebugTest(scripts.Script):
    def title(self):
        return "DEBUG: Denoiser Interceptor (Step 0 Overwrite)"

    def show(self, is_img2img):
        return True

    def ui(self, is_img2img):
        return []

    def run(self, p):
        print("DEBUG: Installing Denoiser Interceptor...")

        # Define the interceptor function
        def step_zero_hijack(params):
            # params.x is the actual latent tensor in the VRAM
            # params.sampling_step tells us where we are
            
            if params.sampling_step == 0:
                print(f"DEBUG: 🛑 Intercepted Step 0! Overwriting {params.x.shape} with Zeros.")
                
                # Force in-place overwrite of the tensor data
                # This changes the memory directly, Forge cannot ignore this.
                params.x.zero_()
                
        # Attach to the processing object
        p.cfg_denoiser_callback = step_zero_hijack
        
        return processing.process_images(p)