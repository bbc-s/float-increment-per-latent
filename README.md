# float-increment-per-latent

## Node for single-pass LoRA sweep
`Per Sample LoRA Loader (Single Pass)`

Inputs:
- `model`
- `lora_name`
- `mode`: `range`, `range_step_by_batch`, `manual_values`
- `start`, `stop`
- `step` (used only in `range`)
- `direction` (used only in `range`)
- `manual_values` (used only in `manual_values`)
- `fallback_adapter_policy`: controls non-fusible adapters such as LoKr

`range_step_by_batch` formula:
- uses the runtime latent batch from `EmptySD3LatentImage`
- if runtime batch is 1: value = `start`
- else: `auto_step = (stop - start) / (runtime_batch - 1)`

Examples:
- start=-4, stop=5, batch_size=10 -> step=1
- start=-10, stop=10, batch_size=10 -> step=2.222222...

Outputs:
- `model` (patched)
- `weights_used` (string preview)

Multiple `Per Sample LoRA Loader (Single Pass)` nodes can be stacked. Stacked LoRAs are aggregated into shared per-layer bypass hooks, and standard LoRA adapters are fused per layer into one larger rank projection where possible.

`fallback_adapter_policy`:
- `error_on_slow_fallback`: stop immediately if any adapter would use the slow runtime path.
- `static_base_only`: keep non-fusible adapters at the base strength and apply per-sample changes only to fusible LoRA adapters.
- `allow_slow_fallback`: exact per-sample behavior for non-fusible adapters, but can be very slow with large stacks.

## Wiring
1. `CheckpointLoaderSimple.model` -> `Per Sample LoRA Loader (Single Pass).model`
2. Output `model` from this node -> `KSampler.model`
3. Set `EmptySD3LatentImage.batch_size` normally; `range_step_by_batch` reads the runtime batch during sampling.
