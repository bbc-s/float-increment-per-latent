import re
from typing import List

import torch
import torch.nn.functional as F

import comfy.lora
import comfy.lora_convert
import comfy.model_management
import comfy.sd
import comfy.utils
import comfy.weight_adapter
from comfy.patcher_extension import PatcherInjection
from comfy.weight_adapter.lora import LoRAAdapter
import folder_paths


AGGREGATE_INJECTION_KEY = "per_sample_lora_aggregate"
AGGREGATE_ENTRIES_KEY = "_per_sample_lora_entries"


class AggregateBypassForwardHook:
    def __init__(self, module, adapters):
        self.module = module
        self.adapters = adapters
        self.original_forward = None
        self.setup_hooks = [
            comfy.weight_adapter.BypassForwardHook(module, adapter, multiplier=1.0)
            for adapter in adapters
        ]
        self.can_fuse_lora = False
        self.fused_down = None
        self.fused_up = None
        self.fused_ranks = None
        self.fused_alpha_scales = None
        self.fused_is_conv = False
        self.fused_conv_dim = 0
        self.fused_kw_dict = {}

    def _forward(self, x, *args, **kwargs):
        out = self.original_forward(x, *args, **kwargs)
        if self.can_fuse_lora:
            return out + self._fused_lora_delta(x, out)
        for adapter in self.adapters:
            out = adapter.g(out + adapter.h(x, out))
        return out

    def _adapter_multiplier(self, adapter, x, out):
        weights_cpu = getattr(adapter, "_pslora_weights_cpu", None)
        dynamic_range = getattr(adapter, "_pslora_dynamic_range", None)
        base_strength = getattr(adapter, "_pslora_base_strength", 0.0)
        actual_batch = int(x.shape[0])

        if dynamic_range is not None:
            runtime_state = getattr(adapter, "_pslora_runtime_state", {})
            logical_batch = int(runtime_state.get("batch_size", actual_batch))
            start, stop = dynamic_range
            weights = PerSampleLoraLoader._build_range_step_by_batch_tensor(
                start, stop, logical_batch, x.device, out.dtype
            )
            deltas = weights - base_strength
            return PerSampleLoraLoader._expand_to_actual_batch(deltas, actual_batch)

        weights = weights_cpu.to(device=x.device, dtype=out.dtype)
        return PerSampleLoraLoader._expand_to_actual_batch(weights, actual_batch)

    def _fused_lora_delta(self, x, out):
        rank_scales = []

        for adapter, rank, alpha_scale in zip(self.adapters, self.fused_ranks, self.fused_alpha_scales):
            sample_scale = self._adapter_multiplier(adapter, x, out).to(dtype=x.dtype) * alpha_scale
            rank_scales.append(sample_scale[:, None].expand(-1, rank))

        if self.fused_is_conv:
            conv_fn = (F.conv1d, F.conv2d, F.conv3d)[self.fused_conv_dim - 1]
            hidden = conv_fn(x, self.fused_down.to(dtype=x.dtype), **self.fused_kw_dict)
            scale = torch.cat(rank_scales, dim=1).view(x.shape[0], -1, *([1] * self.fused_conv_dim))
            hidden = hidden * scale
            return conv_fn(hidden, self.fused_up.to(dtype=x.dtype))

        hidden = F.linear(x, self.fused_down.to(dtype=x.dtype))
        scale_shape = [x.shape[0]] + [1] * (hidden.ndim - 2) + [-1]
        hidden = hidden * torch.cat(rank_scales, dim=1).view(*scale_shape)
        return F.linear(hidden, self.fused_up.to(dtype=x.dtype))

    def _prepare_fused_lora(self, dtype):
        self.fused_is_conv = getattr(self.adapters[0], "is_conv", False)
        self.fused_conv_dim = getattr(self.adapters[0], "conv_dim", 0)
        self.fused_kw_dict = getattr(self.adapters[0], "kw_dict", {})

        down_weights = []
        up_weights = []
        ranks = []
        alpha_scales = []

        for adapter in self.adapters:
            up, down, alpha, _mid, _dora_scale, _reshape = adapter.weights
            if dtype is not None:
                up = up.to(dtype=dtype)
                down = down.to(dtype=dtype)

            rank = int(down.shape[0])
            ranks.append(rank)
            alpha_scales.append((float(alpha) / rank) if alpha is not None else 1.0)

            if self.fused_is_conv:
                kernel_size = getattr(adapter, "kernel_size", (1,) * self.fused_conv_dim)
                in_channels = getattr(adapter, "in_channels", None)
                if down.dim() == 2:
                    if in_channels is not None:
                        down = down.view(down.shape[0], in_channels, *kernel_size)
                    else:
                        down = down.view(*down.shape, *([1] * self.fused_conv_dim))
                if up.dim() == 2:
                    up = up.view(*up.shape, *([1] * self.fused_conv_dim))

            down_weights.append(down)
            up_weights.append(up)

        self.fused_down = torch.cat(down_weights, dim=0).contiguous()
        self.fused_up = torch.cat(up_weights, dim=1).contiguous()
        self.fused_ranks = ranks
        self.fused_alpha_scales = alpha_scales

    def _can_fuse_lora(self):
        if not self.adapters or not all(isinstance(adapter, LoRAAdapter) for adapter in self.adapters):
            return False
        is_conv = getattr(self.adapters[0], "is_conv", False)
        conv_dim = getattr(self.adapters[0], "conv_dim", 0)
        kw_dict = getattr(self.adapters[0], "kw_dict", {})
        for adapter in self.adapters:
            _up, _down, _alpha, mid, dora_scale, reshape = adapter.weights
            if mid is not None or dora_scale is not None or reshape is not None:
                return False
            if getattr(adapter, "is_conv", False) != is_conv:
                return False
            if getattr(adapter, "conv_dim", 0) != conv_dim:
                return False
            if getattr(adapter, "kw_dict", {}) != kw_dict:
                return False
        return True

    def inject(self):
        if self.original_forward is not None:
            return

        device = comfy.model_management.get_torch_device()
        dtype = None
        if hasattr(self.module, "weight") and self.module.weight is not None:
            dtype = self.module.weight.dtype
        if dtype is not None and dtype not in (torch.float32, torch.float16, torch.bfloat16):
            dtype = None

        for setup_hook in self.setup_hooks:
            setup_hook._move_adapter_weights_to_device(device, dtype)

        self.can_fuse_lora = self._can_fuse_lora()
        if self.can_fuse_lora:
            self._prepare_fused_lora(dtype)
        self.original_forward = self.module.forward
        self.module.forward = self._forward

    def eject(self):
        if self.original_forward is None:
            return
        self.module.forward = self.original_forward
        self.original_forward = None
        self.fused_down = None
        self.fused_up = None
        self.fused_ranks = None
        self.fused_alpha_scales = None


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
    def _build_range_step_by_batch_tensor(start: float, stop: float, batch_size: int, device, dtype):
        if batch_size <= 1:
            return torch.tensor([start], device=device, dtype=dtype)
        return torch.linspace(start, stop, steps=batch_size, device=device, dtype=dtype)

    @staticmethod
    def _install_runtime_batch_tracker(model, runtime_state):
        previous_wrapper = model.model_options.get("model_function_wrapper")

        def wrapper(model_apply, args):
            cond_chunks = len(args.get("cond_or_uncond", [])) or 1
            runtime_state["batch_size"] = max(1, int(args["input"].shape[0]) // cond_chunks)
            if previous_wrapper is not None:
                return previous_wrapper(model_apply, args)
            return model_apply(args["input"], args["timestep"], **args["c"])

        model.set_model_unet_function_wrapper(wrapper)

    @staticmethod
    def _get_module_by_key(model, key: str):
        module_key = key[:-7] if key.endswith(".weight") else key
        module = model
        for part in module_key.split("."):
            module = module[int(part)] if part.isdigit() else getattr(module, part)
        return module

    @classmethod
    def _set_aggregate_injection(cls, model):
        entries = model.model_options.get(AGGREGATE_ENTRIES_KEY, [])
        grouped = {}
        for entry in entries:
            grouped.setdefault(entry["key"], []).append(entry["adapter"])

        hooks = []
        for key, adapters in grouped.items():
            try:
                module = cls._get_module_by_key(model.model, key)
            except (AttributeError, IndexError, KeyError):
                continue
            if hasattr(module, "weight"):
                hooks.append(AggregateBypassForwardHook(module, adapters))

        def inject_all(model_patcher):
            for hook in hooks:
                hook.inject()

        def eject_all(model_patcher):
            for hook in hooks:
                hook.eject()

        model.set_injections(AGGREGATE_INJECTION_KEY, [PatcherInjection(inject=inject_all, eject=eject_all)])
        return len(hooks)

    @staticmethod
    def _expand_to_actual_batch(weights: torch.Tensor, actual_batch: int) -> torch.Tensor:
        logical_batch = int(weights.shape[0])
        if logical_batch == actual_batch:
            return weights
        if logical_batch > 0 and actual_batch > logical_batch:
            repeats = (actual_batch + logical_batch - 1) // logical_batch
            return weights.repeat(repeats)[:actual_batch]
        return weights[:actual_batch]

    @staticmethod
    def _attach_per_sample_multiplier(
        adapter,
        per_sample_weights: List[float] | None,
        runtime_state,
        dynamic_range: tuple[float, float] | None,
        base_strength: float,
    ):
        if not hasattr(adapter, "h"):
            return
        original_h = adapter.h
        weights_cpu = None
        if per_sample_weights is not None:
            weights_cpu = torch.tensor(per_sample_weights, dtype=torch.float32)
        adapter._pslora_weights_cpu = weights_cpu
        adapter._pslora_runtime_state = runtime_state
        adapter._pslora_dynamic_range = dynamic_range
        adapter._pslora_base_strength = base_strength

        def patched_h(x, base_out):
            actual_batch = int(x.shape[0])
            if dynamic_range is not None:
                logical_batch = int(runtime_state.get("batch_size", actual_batch))
                start, stop = dynamic_range
                weights = PerSampleLoraLoader._build_range_step_by_batch_tensor(
                    start, stop, logical_batch, x.device, base_out.dtype
                )
                deltas = weights - base_strength
                w_batch = PerSampleLoraLoader._expand_to_actual_batch(deltas, actual_batch)
            else:
                w = weights_cpu.to(device=x.device, dtype=base_out.dtype)
                w_batch = PerSampleLoraLoader._expand_to_actual_batch(w, actual_batch)

            old_multiplier = getattr(adapter, "multiplier", 1.0)
            view_shape = [actual_batch] + [1] * (base_out.ndim - 1)
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
    ):
        if mode == "manual_values":
            values = self._parse_manual_values(manual_values)
        elif mode == "range_step_by_batch":
            values = None
        else:
            values = self._build_range(start, stop, step, direction)
        if values is not None and len(values) == 0:
            raise ValueError("No weights were generated.")
        dynamic_range = None
        if mode == "range_step_by_batch":
            base_strength = float(min(start, stop))
            delta_weights = None
            dynamic_range = (float(start), float(stop))
        else:
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
        runtime_state = {}
        if dynamic_range is not None:
            self._install_runtime_batch_tracker(new_model, runtime_state)

        # Baseline identical to normal LoRA application (model-only):
        # apply full LoRA first using strength = lowest requested weight.
        # Then bypass adds per-sample deltas on top of this baseline.
        new_model.add_patches(loaded, base_strength)

        matched_bypass = 0
        aggregate_entries = new_model.model_options.get(AGGREGATE_ENTRIES_KEY, [])
        if bypass_patches:
            model_sd_keys = set(new_model.model.state_dict().keys())
            for key, adapter in bypass_patches.items():
                if key in model_sd_keys:
                    self._attach_per_sample_multiplier(
                        adapter,
                        delta_weights,
                        runtime_state,
                        dynamic_range,
                        base_strength,
                    )
                    aggregate_entries.append({"key": key, "adapter": adapter})
                    matched_bypass += 1

        new_model.model_options[AGGREGATE_ENTRIES_KEY] = aggregate_entries
        aggregate_hooks = self._set_aggregate_injection(new_model)

        if matched_bypass == 0:
            raise ValueError(
                "Per-sample LoRA hooks were not attached (0 matched adapter keys). "
                "This LoRA/model pair is not compatible with this per-sample bypass method."
            )

        if dynamic_range is not None:
            used = (
                f"weights=runtime_linspace(start={start:.6g}, stop={stop:.6g}, batch=EmptySD3LatentImage) "
                f"base={base_strength:.6g} loaded_total={len(loaded)} bypass={len(bypass_patches)} "
                f"regular={len(regular_patches)} matched_bypass={matched_bypass} aggregate_hooks={aggregate_hooks}"
            )
        else:
            used_weights = ", ".join([f"{v:.6g}" for v in weights])
            used_deltas = ", ".join([f"{v:.6g}" for v in delta_weights])
            used = (
                f"weights=[{used_weights}] deltas=[{used_deltas}] base={base_strength:.6g} "
                f"loaded_total={len(loaded)} bypass={len(bypass_patches)} regular={len(regular_patches)} "
                f"matched_bypass={matched_bypass} aggregate_hooks={aggregate_hooks}"
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
