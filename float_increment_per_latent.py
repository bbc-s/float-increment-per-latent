import re
from typing import List

import torch

import comfy.lora
import comfy.lora_convert
import comfy.sd
import comfy.utils
import comfy.weight_adapter
import folder_paths


class FloatIncrementPerLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["range", "manual_values"], {"default": "range"}),
                "start": (
                    "FLOAT",
                    {"default": 0.0, "min": -1000000.0, "max": 1000000.0, "step": 0.01},
                ),
                "stop": (
                    "FLOAT",
                    {"default": 1.0, "min": -1000000.0, "max": 1000000.0, "step": 0.01},
                ),
                "step": (
                    "FLOAT",
                    {"default": 0.1, "min": 0.000001, "max": 1000000.0, "step": 0.01},
                ),
                "direction": (["increment", "decrement"], {"default": "increment"}),
                "manual_values": (
                    "STRING",
                    {
                        "default": "-2, -0.5, 0, 0.5, 1",
                        "multiline": True,
                    },
                ),
                "round_to": ("INT", {"default": 6, "min": 0, "max": 12, "step": 1}),
            }
        }

    RETURN_TYPES = ("FLOAT", "INT")
    RETURN_NAMES = ("value", "index")
    OUTPUT_IS_LIST = (True, True)
    FUNCTION = "build"
    CATEGORY = "utils/float"

    @staticmethod
    def _round_value(value: float, decimals: int) -> float:
        return round(value, decimals)

    @staticmethod
    def _parse_manual_values(text: str) -> List[float]:
        parts = [p.strip() for p in re.split(r"[,;\n\t ]+", text) if p.strip()]
        return [float(p) for p in parts]

    def _build_range(self, start: float, stop: float, step: float, direction: str) -> List[float]:
        if step <= 0:
            raise ValueError("step must be > 0")

        signed_step = abs(step)
        if direction == "decrement":
            signed_step = -signed_step

        values: List[float] = []
        current = start

        if signed_step > 0:
            while current <= stop + 1e-12:
                values.append(current)
                current += signed_step
        else:
            while current >= stop - 1e-12:
                values.append(current)
                current += signed_step

        if not values:
            raise ValueError(
                "Range produced 0 values. Check start/stop/direction (e.g. increment requires start <= stop)."
            )

        return values

    def build(self, mode, start, stop, step, direction, manual_values, round_to):
        if mode == "manual_values":
            values = self._parse_manual_values(manual_values)
            if not values:
                raise ValueError("manual_values must contain at least one number.")
        else:
            values = self._build_range(start, stop, step, direction)

        out_values = [self._round_value(v, round_to) for v in values]
        out_indices = list(range(len(out_values)))
        return (out_values, out_indices)


class PerSampleLoraLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "mode": (["range", "range_step_by_batch", "manual_values"], {"default": "range"}),
                "start": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "stop": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "step": ("FLOAT", {"default": 0.1, "min": 0.000001, "max": 100.0, "step": 0.01}),
                "direction": (["increment", "decrement"], {"default": "increment"}),
                "manual_values": ("STRING", {"default": "0.1,0.2,0.3,0.4", "multiline": True}),
                "batch_size": ("INT", {"default": 10, "min": 1, "max": 8192, "step": 1}),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "weights_used")
    FUNCTION = "load_per_sample_lora"
    CATEGORY = "loaders"

    @staticmethod
    def _parse_manual_values(text: str) -> List[float]:
        parts = [p.strip() for p in re.split(r"[,;\n\t ]+", text) if p.strip()]
        return [float(p) for p in parts]

    @staticmethod
    def _build_range(start: float, stop: float, step: float, direction: str) -> List[float]:
        step_abs = abs(step)
        if step_abs <= 0:
            raise ValueError("step must be > 0")

        lo = min(start, stop)
        hi = max(start, stop)
        values: List[float] = []

        if direction == "increment":
            cur = lo
            while cur <= hi + 1e-12:
                values.append(cur)
                cur += step_abs
        else:
            cur = hi
            while cur >= lo - 1e-12:
                values.append(cur)
                cur -= step_abs
        return values

    @staticmethod
    def _build_range_step_by_batch(start: float, stop: float, batch_size: int) -> List[float]:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if batch_size == 1:
            return [start]
        auto_step = (stop - start) / float(batch_size - 1)
        return [start + (auto_step * i) for i in range(batch_size)]

    @staticmethod
    def _attach_per_sample_multiplier(adapter, per_sample_weights: List[float]):
        if not hasattr(adapter, "h"):
            return
        original_h = adapter.h
        weights_cpu = torch.tensor(per_sample_weights, dtype=torch.float32)

        def patched_h(x, base_out):
            batch = x.shape[0]
            w = weights_cpu.to(device=x.device, dtype=base_out.dtype)
            n = w.shape[0]
            if batch == n:
                w_batch = w
            elif n > 0 and batch > n:
                repeats = (batch + n - 1) // n
                w_batch = w.repeat(repeats)[:batch]
            else:
                w_batch = w[:batch]

            old_multiplier = getattr(adapter, "multiplier", 1.0)
            view_shape = [batch] + [1] * (base_out.ndim - 1)
            adapter.multiplier = w_batch.view(*view_shape)
            try:
                return original_h(x, base_out)
            finally:
                adapter.multiplier = old_multiplier

        adapter.h = patched_h

    def load_per_sample_lora(
        self,
        model,
        lora_name,
        mode,
        start,
        stop,
        step,
        direction,
        manual_values,
        batch_size,
    ):
        if mode == "manual_values":
            values = self._parse_manual_values(manual_values)
        elif mode == "range_step_by_batch":
            values = self._build_range_step_by_batch(start, stop, batch_size)
        else:
            values = self._build_range(start, stop, step, direction)
        if len(values) == 0:
            raise ValueError("No weights were generated.")
        weights = values
        base_strength = float(min(weights))
        delta_weights = [float(w - base_strength) for w in weights]

        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora_file = comfy.utils.load_torch_file(lora_path, safe_load=True)

        key_map = comfy.lora.model_lora_keys_unet(model.model, {})

        converted = comfy.lora_convert.convert_lora(lora_file)
        loaded = comfy.lora.load_lora(converted, key_map)

        bypass_patches = {}
        regular_patches = {}
        for key, patch_data in loaded.items():
            if isinstance(patch_data, comfy.weight_adapter.WeightAdapterBase):
                bypass_patches[key] = patch_data
            else:
                regular_patches[key] = patch_data

        new_model = model.clone()
        # Baseline identical to normal LoRA application (model-only):
        # apply full LoRA first using strength = first requested weight.
        # Then bypass adds per-sample deltas on top of this baseline.
        new_model.add_patches(loaded, base_strength)

        matched_bypass = 0
        if bypass_patches:
            model_manager = comfy.weight_adapter.BypassInjectionManager()
            model_sd_keys = set(new_model.model.state_dict().keys())
            for key, adapter in bypass_patches.items():
                if key in model_sd_keys:
                    self._attach_per_sample_multiplier(adapter, delta_weights)
                    model_manager.add_adapter(key, adapter, strength=1.0)
                    matched_bypass += 1
            model_injections = model_manager.create_injections(new_model.model)
            if model_manager.get_hook_count() > 0:
                new_model.set_injections("per_sample_lora", model_injections)

        if matched_bypass == 0:
            raise ValueError(
                "Per-sample LoRA hooks were not attached (0 matched adapter keys). "
                "This LoRA/model pair is not compatible with this per-sample bypass method."
            )

        used_weights = ", ".join([f"{v:.6g}" for v in weights])
        used_deltas = ", ".join([f"{v:.6g}" for v in delta_weights])
        used = (
            f"weights=[{used_weights}] deltas=[{used_deltas}] base={base_strength:.6g} "
            f"loaded_total={len(loaded)} bypass={len(bypass_patches)} regular={len(regular_patches)} "
            f"matched_bypass={matched_bypass}"
        )
        return (new_model, used)


NODE_CLASS_MAPPINGS = {
    "FloatIncrementPerLatent": FloatIncrementPerLatent,
    "PerSampleLoraLoader": PerSampleLoraLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FloatIncrementPerLatent": "Float Increment Per Latent",
    "PerSampleLoraLoader": "Per Sample LoRA Loader (Single Pass)",
}
