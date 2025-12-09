function extract_tags_to_txt2img(tags) {
    // 1. Switch to the txt2img tab
    switch_to_txt2img();

    // 2. Find the prompt textarea (Works for Forge & A1111)
    const prompt_box = gradioApp().querySelector('#txt2img_prompt textarea');
    
    if (prompt_box && tags) {
        // 3. Append tags safely
        const current_text = prompt_box.value;
        const separator = current_text.length > 0 && !current_text.endsWith(", ") ? ", " : "";
        prompt_box.value = current_text + separator + tags;
        
        // 4. Update Gradio (Important!)
        prompt_box.dispatchEvent(new Event("input", { bubbles: true }));
    }
    
    return [];
}