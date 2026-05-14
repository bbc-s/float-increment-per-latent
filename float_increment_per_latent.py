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
        self.fused_adapters = []
        self.fused_lokr_adapters = []
        self.fallback_adapters = adapters[:]
        self.fused_down = None
        self.fused_up = None
        self.fused_ranks = None
        self.fused_alpha_scales = None
        self.fused_is_conv = False
        self.fused_conv_dim = 0
        self.fused_kw_dict = {}
        self.fused_lokr_groups = []

    def _forward(self, x, *args, **kwargs):
        out = self.original_forward(x, *args, **kwargs)
        if self.fused_adapters:
            out = out + self._fused_lora_delta(x, out)
        for group in self.fused_lokr_groups:
            out = out + self._fused_lokr_delta(x, out, group)
        for adapter in self.fallback_adapters:
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

        for adapter, rank, alpha_scale in zip(self.fused_adapters, self.fused_ranks, self.fused_alpha_scales):
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

    def _fused_lokr_delta(self, x, out, group):
        adapters = group["adapters"]
        scales = torch.stack([
            self._adapter_multiplier(adapter, x, out).to(dtype=x.dtype)
            for adapter in adapters
        ], dim=1)

        if group["is_conv"]:
            conv_fn = (F.conv1d, F.conv2d, F.conv3d)[group["conv_dim"] - 1]
            b, _c, *spatial = x.shape
            uq = group["uq"]
            h_in_group = x.reshape(b * uq, -1, *spatial)
            hb = conv_fn(h_in_group, group["w2"].to(dtype=x.dtype), **group["kw_dict"])
            out_k = group["out_k"]
            spatial_out = hb.shape[2:]
            hb = hb.view(b, uq, len(adapters), out_k, *spatial_out)
            hc = torch.einsum("bqnk...,nlq->bnlk...", hb, group["w1"].to(dtype=x.dtype))
            hc = hc * scales.view(b, len(adapters), 1, 1, *([1] * len(spatial_out)))
            return hc.sum(dim=1).reshape(b, -1, *spatial_out)

        uq = group["uq"]
        h_in_group = x.reshape(*x.shape[:-1], uq, -1)
        hb = F.linear(h_in_group, group["w2"].to(dtype=x.dtype))
        out_k = group["out_k"]
        hb = hb.view(*h_in_group.shape[:-1], len(adapters), out_k)
        hc = torch.einsum("...qnk,nlq->...nlk", hb, group["w1"].to(dtype=x.dtype))
        hc = hc * scales.view(*([x.shape[0]] + [1] * (hc.ndim - 4) + [len(adapters), 1, 1]))
        return hc.sum(dim=-3).reshape(*x.shape[:-1], -1)

    def _prepare_fused_lora(self, dtype):
        self.fused_is_conv = getattr(self.fused_adapters[0], "is_conv", False)
        self.fused_conv_dim = getattr(self.fused_adapters[0], "conv_dim", 0)
        self.fused_kw_dict = getattr(self.fused_adapters[0], "kw_dict", {})

        down_weights = []
        up_weights = []
        ranks = []
        alpha_scales = []

        for adapter in self.fused_adapters:
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

    def _prepare_fused_lokr(self, dtype):
        groups = {}
        for adapter in self.fused_lokr_adapters:
            if not self._is_lokr_direct_like(adapter):
                continue
            w1, w2 = adapter.weights[0], adapter.weights[1]
            key = (
                tuple(w1.shape),
                tuple(w2.shape),
                getattr(adapter, "is_conv", False),
                getattr(adapter, "conv_dim", 0),
                tuple(sorted(getattr(adapter, "kw_dict", {}).items())),
            )
            groups.setdefault(key, []).append(adapter)

        self.fused_lokr_groups = []
        for adapters in groups.values():
            first = adapters[0]
            is_conv = getattr(first, "is_conv", False)
            conv_dim = getattr(first, "conv_dim", 0)
            w1_list = []
            w2_list = []
            for adapter in adapters:
                w1, w2 = adapter.weights[0], adapter.weights[1]
                if dtype is not None:
                    w1 = w1.to(dtype=dtype)
                    w2 = w2.to(dtype=dtype)
                if is_conv and w2.dim() == 2:
                    w2 = w2.view(*w2.shape, *([1] * conv_dim))
                w1_list.append(w1)
                w2_list.append(w2)

            self.fused_lokr_groups.append({
                "adapters": adapters,
                "w1": torch.stack(w1_list, dim=0).contiguous(),
                "w2": torch.cat(w2_list, dim=0).contiguous(),
                "uq": int(w1_list[0].shape[1]),
                "out_k": int(w2_list[0].shape[0]),
                "is_conv": is_conv,
                "conv_dim": conv_dim,
                "kw_dict": getattr(first, "kw_dict", {}) if is_conv else {},
            })

    def _can_fuse_lora(self):
        fused, _fallback, _reasons = self._partition_adapters_for_fusion()
        return len(fused) > 0

    @staticmethod
    def _is_lora_like(adapter):
        weights = getattr(adapter, "weights", None)
        if not isinstance(weights, (list, tuple)) or len(weights) != 6:
            return False
        up, down, _alpha, _mid, _dora_scale, _reshape = weights
        return (
            hasattr(up, "shape")
            and hasattr(up, "to")
            and hasattr(down, "shape")
            and hasattr(down, "to")
        )

    @staticmethod
    def _is_lokr_direct_like(adapter):
        weights = getattr(adapter, "weights", None)
        if type(adapter).__name__ != "LoKrAdapter":
            return False
        if not isinstance(weights, (list, tuple)) or len(weights) != 9:
            return False
        w1, w2 = weights[0], weights[1]
        if w1 is None or w2 is None:
            return False
        if any(weights[index] is not None for index in (3, 4, 5, 6, 7)):
            return False
        return hasattr(w1, "shape") and hasattr(w1, "to") and hasattr(w2, "shape") and hasattr(w2, "to")

    @staticmethod
    def _adapter_debug_name(adapter):
        weights = getattr(adapter, "weights", None)
        if not isinstance(weights, (list, tuple)):
            return f"{type(adapter).__name__}:weights={type(weights).__name__}"
        shapes = []
        for item in weights[:2]:
            shapes.append("x".join(str(dim) for dim in item.shape) if isinstance(item, torch.Tensor) else type(item).__name__)
        return f"{type(adapter).__name__}:len={len(weights)}:w0={shapes[0]}:w1={shapes[1]}"

    def _adapter_fuse_blocker_reason(self, adapter, ref_adapter):
        if not self._is_lora_like(adapter):
            return "non_lora_adapter"
        _up, _down, _alpha, mid, _dora_scale, _reshape = adapter.weights
        ref_up, ref_down = ref_adapter.weights[0], ref_adapter.weights[1]
        if _up.shape[0] != ref_up.shape[0] or _down.shape[1:] != ref_down.shape[1:]:
            return "shape_mismatch"
        # Bypass LoRA h() ignores dora_scale and reshape metadata. The fused
        # path mirrors that bypass behavior and only rejects LoCon mid weights.
        if mid is not None:
            return "mid_weights"
        if getattr(adapter, "is_conv", False) != getattr(ref_adapter, "is_conv", False):
            return "conv_type_mismatch"
        if getattr(adapter, "conv_dim", 0) != getattr(ref_adapter, "conv_dim", 0):
            return "conv_dim_mismatch"
        if getattr(adapter, "kw_dict", {}) != getattr(ref_adapter, "kw_dict", {}):
            return "kw_dict_mismatch"
        return None

    def _partition_adapters_for_fusion(self):
        ref_adapter = next((adapter for adapter in self.adapters if self._is_lora_like(adapter)), None)
        if ref_adapter is None:
            fused = [adapter for adapter in self.adapters if self._is_lokr_direct_like(adapter)]
            fallback = [adapter for adapter in self.adapters if not self._is_lokr_direct_like(adapter)]
            reasons = {}
            if fallback:
                reasons["non_lora_adapter"] = len(fallback)
            return fused, fallback, reasons

        fused = []
        fallback = []
        reasons = {}
        for adapter in self.adapters:
            if self._is_lokr_direct_like(adapter):
                fused.append(adapter)
                continue
            reason = self._adapter_fuse_blocker_reason(adapter, ref_adapter)
            if reason is None:
                fused.append(adapter)
            else:
                fallback.append(adapter)
                reasons[reason] = reasons.get(reason, 0) + 1
        return fused, fallback, reasons

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

        fused_adapters, self.fallback_adapters, _reasons = self._partition_adapters_for_fusion()
        self.fused_adapters = [adapter for adapter in fused_adapters if self._is_lora_like(adapter)]
        self.fused_lokr_adapters = [adapter for adapter in fused_adapters if self._is_lokr_direct_like(adapter)]
        if self.fused_adapters:
            self._prepare_fused_lora(dtype)
        self._prepare_fused_lokr(dtype)
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
        self.fused_adapters = []
        self.fused_lokr_adapters = []
        self.fallback_adapters = self.adapters[:]
        self.fused_lokr_groups = []


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
                "fallback_adapter_policy": (
                    ["error_on_slow_fallback", "static_base_only", "allow_slow_fallback"],
                    {"default": "error_on_slow_fallback"},
                ),
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
        fused_possible = 0
        fallback_hooks = 0
        fallback_reasons = {}
        fallback_types = {}
        for key, adapters in grouped.items():
            try:
                module = cls._get_module_by_key(model.model, key)
            except (AttributeError, IndexError, KeyError):
                continue
            if hasattr(module, "weight"):
                hook = AggregateBypassForwardHook(module, adapters)
                fused, fallback, reasons = hook._partition_adapters_for_fusion()
                if fused:
                    fused_possible += 1
                if fallback:
                    fallback_hooks += 1
                    for reason, count in reasons.items():
                        fallback_reasons[reason] = fallback_reasons.get(reason, 0) + count
                    for adapter in fallback:
                        if not hook._is_lora_like(adapter):
                            type_name = hook._adapter_debug_name(adapter)
                            fallback_types[type_name] = fallback_types.get(type_name, 0) + 1
                hooks.append(hook)

        def inject_all(model_patcher):
            for hook in hooks:
                hook.inject()

        def eject_all(model_patcher):
            for hook in hooks:
                hook.eject()

        model.set_injections(AGGREGATE_INJECTION_KEY, [PatcherInjection(inject=inject_all, eject=eject_all)])
        return {
            "hooks": len(hooks),
            "entries": len(entries),
            "fused_possible": fused_possible,
            "fallback_hooks": fallback_hooks,
            "fallback_reasons": fallback_reasons,
            "fallback_types": fallback_types,
        }

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
        fallback_adapter_policy,
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
        skipped_static_adapters = 0
        aggregate_entries = new_model.model_options.get(AGGREGATE_ENTRIES_KEY, [])
        if bypass_patches:
            model_sd_keys = set(new_model.model.state_dict().keys())
            for key, adapter in bypass_patches.items():
                if key in model_sd_keys:
                    if (
                        fallback_adapter_policy == "static_base_only"
                        and not AggregateBypassForwardHook._is_lora_like(adapter)
                        and not AggregateBypassForwardHook._is_lokr_direct_like(adapter)
                    ):
                        skipped_static_adapters += 1
                        continue
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
        aggregate_stats = self._set_aggregate_injection(new_model)
        fallback_reasons = ",".join(
            f"{key}:{value}" for key, value in sorted(aggregate_stats["fallback_reasons"].items())
        ) or "none"
        fallback_types = ";".join(
            f"{key}:{value}" for key, value in sorted(aggregate_stats["fallback_types"].items())
        ) or "none"

        if matched_bypass == 0:
            raise ValueError(
                "Per-sample LoRA hooks were not attached (0 matched adapter keys). "
                "This LoRA/model pair is not compatible with this per-sample bypass method."
            )
        if fallback_adapter_policy == "error_on_slow_fallback" and aggregate_stats["fallback_hooks"] > 0:
            raise ValueError(
                "Slow per-sample fallback adapters detected. "
                f"fallback_reasons={fallback_reasons}; fallback_types={fallback_types}. "
                "Use fallback_adapter_policy=static_base_only to keep these adapters at base strength, "
                "or allow_slow_fallback to run the slow exact path."
            )

        if dynamic_range is not None:
            used = (
                f"weights=runtime_linspace(start={start:.6g}, stop={stop:.6g}, batch=EmptySD3LatentImage) "
                f"base={base_strength:.6g} loaded_total={len(loaded)} bypass={len(bypass_patches)} "
                f"regular={len(regular_patches)} matched_bypass={matched_bypass} "
                f"aggregate_entries={aggregate_stats['entries']} aggregate_hooks={aggregate_stats['hooks']} "
                f"fused_hooks={aggregate_stats['fused_possible']} fallback_hooks={aggregate_stats['fallback_hooks']} "
                f"fallback_reasons={fallback_reasons} fallback_types={fallback_types} "
                f"skipped_static_adapters={skipped_static_adapters}"
            )
        else:
            used_weights = ", ".join([f"{v:.6g}" for v in weights])
            used_deltas = ", ".join([f"{v:.6g}" for v in delta_weights])
            used = (
                f"weights=[{used_weights}] deltas=[{used_deltas}] base={base_strength:.6g} "
                f"loaded_total={len(loaded)} bypass={len(bypass_patches)} regular={len(regular_patches)} "
                f"matched_bypass={matched_bypass} "
                f"aggregate_entries={aggregate_stats['entries']} aggregate_hooks={aggregate_stats['hooks']} "
                f"fused_hooks={aggregate_stats['fused_possible']} fallback_hooks={aggregate_stats['fallback_hooks']} "
                f"fallback_reasons={fallback_reasons} fallback_types={fallback_types} "
                f"skipped_static_adapters={skipped_static_adapters}"
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
