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
- `batch_size` (used only in `range_step_by_batch`)

`range_step_by_batch` formula:
- if `batch_size = 1`: value = `start`
- else: `auto_step = (stop - start) / (batch_size - 1)`

Examples:
- start=-4, stop=5, batch_size=10 -> step=1
- start=-10, stop=10, batch_size=10 -> step=2.222222...

Outputs:
- `model` (patched)
- `weights_used` (string preview)

## Wiring
1. `CheckpointLoaderSimple.model` -> `Per Sample LoRA Loader (Single Pass).model`
2. Output `model` from this node -> `KSampler.model`
3. Keep `EmptySD3LatentImage.batch_size` aligned with this node's `batch_size` when using `range_step_by_batch`.
