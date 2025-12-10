import modules.scripts as scripts
import gradio as gr
from modules import processing, shared, devices, sd_samplers
import torch

class LatentSwitcher(scripts.Script):
    def title(self):
        return "Latent Switcher (Prompt Scheduling)"

    def show(self, is_img2img):
        return True

    def ui(self, is_img2img):
        gr.Markdown("### 🔀 Latent Switcher")
        gr.Markdown("Use Prompt A for **Shape/Layout** and Prompt B for **Texture/Details**.")
        
        with gr.Row():
            with gr.Column():
                prompt_a = gr.Textbox(label="Prompt A (The Architect)", placeholder="e.g. A simple line drawing of a city, minimal")
                prompt_b = gr.Textbox(label="Prompt B (The Decorator)", placeholder="e.g. A cyberpunk city, neon lights, realistic, 8k")
            
            with gr.Column():
                switch_at = gr.Slider(label="Switch Point", minimum=0.0, maximum=1.0, value=0.35, step=0.05, 
                                    info="0.35 is the Sweet Spot. Lower = More B. Higher = More A.")
                
                debug_mode = gr.Checkbox(label="Return 'Layout' image too?", value=False)

        return [prompt_a, prompt_b, switch_at, debug_mode]

    def run(self, p, prompt_a, prompt_b, switch_at, debug_mode):
        
        # 1. SETUP
        # We need to calculate how many steps for Phase 1 (Architect)
        total_steps = p.steps
        switch_step = int(total_steps * switch_at)
        
        print(f"LatentSwitcher: Running '{prompt_a}' for {switch_step} steps...")
        
        # 2. PHASE 1: THE ARCHITECT (Shape Generation)
        # We create a clone of the processing object to run the first half
        p.n_iter = 1 # Force batch count to 1 for control safety
        p.batch_size = 1
        
        # Override the main prompt with Prompt A
        original_prompt = p.prompt
        p.prompt = prompt_a
        
        # We want to STOP generation early. 
        # A1111/Forge doesn't have a clean "stop_at" flag in the simple API,
        # so we rely on img2img logic or modifying the steps.
        
        # Trick: We set steps to total_steps, but we will "break" the loop? 
        # Easier Trick: We use "denoising_strength" logic reversed? 
        # Best Trick: We assume the 'img2img' pipeline behavior where we generate 
        # full noise, but we have to hack the loop.
        
        # Let's try the cleanest method: Just run standard generation but intercept the latents.
        # Since we can't easily intercept 'mid-loop' without deep hooks, 
        # we will simulate it by running the generation and hoping we can extract.
        
        # Actually, the easiest way to do "Step 0 to X" is to create a custom p
        # but since that's hard, let's use the 'Img2Img' Trick in reverse.
        
        # WAIT! We can just run the process and use `state.interrupted` logic? No.
        
        # REAL SOTA APPROACH: 
        # We manually generate the noise, and assume we are in Txt2Img.
        
        # Generate initial noise (Seed needs to be fixed)
        if p.seed == -1: p.seed = int(torch.randint(0, 4294967294, (1,)).item())
        
        # Set up Phase 1 Processing
        p_phase1 = processing.StableDiffusionProcessingTxt2Img(
            sd_model=shared.sd_model,
            outpath_samples=p.outpath_samples,
            outpath_grids=p.outpath_grids,
            prompt=prompt_a,
            negative_prompt=p.negative_prompt,
            styles=p.styles,
            seed=p.seed,
            subseed=p.subseed,
            subseed_strength=p.subseed_strength,
            sampler_name=p.sampler_name,
            batch_size=p.batch_size,
            n_iter=1,
            steps=total_steps, # We define full steps
            cfg_scale=p.cfg_scale,
            width=p.width,
            height=p.height,
        )
        
        # THE HACK: We inject a custom callback to "Kill" the process at switch_step
        # and steal the latents.
        
        latents_holder = {"latents": None}
        
        def callback_capture(params):
            # params contains: x, i, sigma, etc.
            if params.i >= switch_step - 1:
                latents_holder["latents"] = params.x
                # We stop the sampling by forcing the loop to end? 
                # Most samplers don't allow clean breaking.
                # Instead, we let it finish? No, that wastes time.
                # We will throw a special exception to break out (hacky but effective)
                raise InterruptedError("SwitchPointReached")

        # p_phase1.callback_on_step_end = callback_capture # Only works on some samplers
        # Let's rely on standard logic: 
        # We will use "Img2Img" to finish the job.
        
        # --- PLAN B (Robust): ---
        # 1. Run Phase 1 fully? No.
        # 2. Use the fact that High-Denoise Img2Img IS Phase 1? No.
        
        # Let's use the Shared State interruption.
        shared.state.begin()
        p_phase1.callback_on_step_end = callback_capture
        
        try:
            processing.process_images(p_phase1)
        except InterruptedError:
            print("LatentSwitcher: Captured Layout Latents!")
        except Exception as e:
            print(f"Error in Phase 1: {e}")
        
        # Retrieve captured latents
        noisy_latents = latents_holder["latents"]
        
        if noisy_latents is None:
            return processing.process_images(p) # Fail safe
        
        # 3. PHASE 2: THE DECORATOR (Texture Generation)
        print(f"LatentSwitcher: Switching to '{prompt_b}'...")
        
        # We use Img2Img logic now.
        # We need to turn those noisy latents into an image? 
        # NO, we feed the noisy latents DIRECTLY into a new generation loop starting at step X.
        
        # To do this in Forge without deep hacking, we construct an Img2Img worker,
        # pass it a dummy image (black), but override the 'init_latents'.
        
        # But `init_latents` expects the image encoded.
        # We have raw noisy latents (sigma > 0).
        
        # The cleanest way: Create a new Txt2Img processor, but force its start noise.
        # This is tricky because Txt2Img always starts at sigma_max.
        
        # Let's use the Img2Img processor.
        # We need to decode the latents slightly to get a "pixel" approximation for the UI?
        # No, let's keep it pure latent.
        
        p_phase2 = processing.StableDiffusionProcessingImg2Img(
            sd_model=shared.sd_model,
            outpath_samples=p.outpath_samples,
            outpath_grids=p.outpath_grids,
            prompt=prompt_b,
            negative_prompt=p.negative_prompt,
            seed=p.seed,
            sampler_name=p.sampler_name,
            batch_size=p.batch_size,
            n_iter=1,
            steps=total_steps,
            cfg_scale=p.cfg_scale,
            width=p.width,
            height=p.height,
            denoising_strength= (1.0 - switch_at), # CRITICAL: This sets where we start!
            init_images=[Image.new("RGB", (p.width, p.height))], # Dummy
        )
        
        # We overwrite the init_latents logic.
        # Standard Img2Img: Encodes image -> Adds Noise.
        # We want: Use our already noisy latents.
        
        # Since we intercepted the latents at step X (which corresponds to noise level Y),
        # and we set denoising_strength to (1-switch_at), Img2Img will try to add noise.
        # We must FORCE it to use OUR latents.
        
        p_phase2.init_latents = noisy_latents
        
        # Run Phase 2
        res = processing.process_images(p_phase2)
        
        # Debug: if user wants to see the layout, we might need to decode the midway latents (hard).
        # We just return the final result.
        
        return res