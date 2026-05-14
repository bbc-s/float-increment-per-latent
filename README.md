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

Multiple `Per Sample LoRA Loader (Single Pass)` nodes can be stacked. Each node keeps its own LoRA bypass hooks, so different LoRAs and ranges can be applied in sequence.

## Wiring
1. `CheckpointLoaderSimple.model` -> `Per Sample LoRA Loader (Single Pass).model`
2. Output `model` from this node -> `KSampler.model`
3. Set `EmptySD3LatentImage.batch_size` normally; `range_step_by_batch` reads the runtime batch during sampling.
