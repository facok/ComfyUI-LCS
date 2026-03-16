import { app } from "../../../scripts/app.js";

// Preset values — the single source of truth for preset parameters.
// TONE_PRESET_NAMES in nodes/intervene.py must list matching keys.
const TONE_PRESETS = {
    "Cinematic":   { contrast: 1.20, brightness: -0.05, saturation: 0.90, color_temperature: 0.15 },
    "HDR":         { contrast: 1.40, brightness:  0.0,  saturation: 1.20, color_temperature: 0.0 },
    "Vivid":       { contrast: 1.10, brightness:  0.0,  saturation: 1.50, color_temperature: 0.0 },
    "Dramatic":    { contrast: 1.50, brightness: -0.10, saturation: 0.85, color_temperature: 0.0 },
    "Low Key":     { contrast: 1.30, brightness: -0.20, saturation: 0.80, color_temperature: 0.0 },
    "High Key":    { contrast: 0.80, brightness:  0.20, saturation: 0.90, color_temperature: 0.0 },
    "Warm":        { contrast: 1.0,  brightness:  0.0,  saturation: 1.0,  color_temperature: 0.50 },
    "Cool":        { contrast: 1.0,  brightness:  0.0,  saturation: 1.0,  color_temperature: -0.50 },
    "Desaturated": { contrast: 1.0,  brightness:  0.0,  saturation: 0.40, color_temperature: 0.0 },
};

const SLIDER_NAMES = ["contrast", "brightness", "saturation", "color_temperature"];

function syncPreset(node, presetName) {
    const preset = TONE_PRESETS[presetName];
    if (!preset) return; // "Custom" — leave sliders as-is

    for (const name of SLIDER_NAMES) {
        const widget = node.widgets?.find(w => w.name === name);
        if (widget && name in preset) {
            widget.value = preset[name];
        }
    }
    node.graph?.setDirtyCanvas(true, true);
}

app.registerExtension({
    name: "ComfyUI-LCS.TonePresetSync",

    nodeCreated(node) {
        if (node.comfyClass !== "LCSToneAdjust") return;

        const presetWidget = node.widgets?.find(w => w.name === "preset");
        if (!presetWidget) return;

        // Only sync on explicit user interaction (dropdown change), not on
        // workflow load or paste — those restore saved slider values directly.
        const origCallback = presetWidget.callback;
        presetWidget.callback = function (value, canvas, nodeRef, pos, event) {
            if (origCallback) origCallback.call(this, value, canvas, nodeRef, pos, event);
            syncPreset(node, value);
        };
    },
});
