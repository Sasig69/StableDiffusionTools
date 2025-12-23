function neoGallerySendToTxt(rawInfo) {
    if (!rawInfo || rawInfo.length < 10) return [];

    // Switch Tab
    const tabs = document.querySelector('#tabs');
    if (tabs) tabs.querySelector('button').click();

    const parts = rawInfo.split("Negative prompt:");
    const positive = parts[0].split("Steps:")[0].trim();
    let negative = "";
    if (rawInfo.includes("Negative prompt:")) {
        negative = parts[1].split("Steps:")[0].trim();
    }

    const extract = (regex) => {
        const m = rawInfo.match(regex);
        return m ? m[1].trim() : null;
    };

    const sets = {
        "#txt2img_prompt textarea": positive,
        "#txt2img_neg_prompt textarea": negative,
        "#txt2img_steps": extract(/Steps:\s*([^,]+)/),
        "#txt2img_sampling": extract(/Sampler:\s*([^,]+)/),
        "#txt2img_cfg_scale": extract(/CFG scale:\s*([^,]+)/),
        "#txt2img_seed": extract(/Seed:\s*([^,]+)/)
    };

    Object.entries(sets).forEach(([selector, val]) => {
        const el = document.querySelector(selector);
        if (el && val) {
            const input = el.tagName === 'TEXTAREA' ? el : el.querySelector('input') || el;
            input.value = val;
            input.dispatchEvent(new Event("input", { bubbles: true }));
        }
    });

    window.scrollTo({ top: 0, behavior: 'smooth' });
    return [];
}