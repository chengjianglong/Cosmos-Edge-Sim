# Cosmos-Edge-Sim

Run NVIDIA's [Cosmos3-Edge](https://huggingface.co/nvidia/Cosmos3-Edge) (4B world model) image-to-video
generation on a **single 6 GB GPU** (tested on an NVIDIA RTX A2000, Ampere sm_86).

The official recipe (`pipe.to("cuda")`) needs ~10 GB of VRAM for the bf16 weights alone.
`lowvram/run_edge.py` brings the peak down to **~3 GB**:

| Technique | What it saves |
|---|---|
| Diffusers group offloading: transformer weights stay in pinned CPU RAM, 2 decoder layers at a time are streamed to the GPU on a side CUDA stream | ~6.7 GB of transformer weights |
| VAE (Wan2.2) kept on CPU; moved to the GPU only for the conditioning encode and the final tiled decode | ~1.4 GB during denoising |
| Image-to-video encodes only the conditioning frame (the stock pipeline VAE-encodes the image repeated over all frames, but keeps only latent frame 0) | ~2.5 GB peak at 480p |
| `cosmos_guardrail` safety checker off by default (`--safety` turns it on) | several GB |

## Results (RTX A2000 6 GB, 30 GB system RAM)

| Frames | Resolution | Steps | Denoise | VAE decode | Total | Peak VRAM |
|---|---|---|---|---|---|---|
| 33 | 832×480 | 20 | 92 s | 26 s | 132 s | 2.2 GB |
| 121 (model-card default) | 832×480 | 20 | 217 s | 94 s | 321 s | 3.0 GB |

## Setup

```bash
scripts/setup_env.sh            # conda env "cosmosim": Python 3.12, torch 2.11 cu128, diffusers 0.41.0
conda activate cosmosim
scripts/download_edge.sh        # ~8.6 GB into checkpoints/Cosmos3-Edge
```

## Usage

```bash
# Model-card example (driving on a coastal road), 121 frames at 480p on GPU 0
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python lowvram/run_edge.py --gpu 0

# Your own conditioning image and prompt (text, .txt, or upsampled .json prompt)
python lowvram/run_edge.py --gpu 1 --image my_frame.jpg --prompt "Dashcam view, light rain, ..." \
    --num-frames 61 -o outputs/my_run.mp4
```

Useful flags:
- `--offload-blocks N`: layers streamed per group. Higher is faster but uses more VRAM.
- `--num-frames`: the model card supports 50–150 frames; 33 also worked.
- `--steps`, `--guidance`, `--flow-shift`, `--seed`.

## Notes and known limits

- **Splitting across two GPUs doesn't work yet.** An `accelerate` `device_map` that splits the decoder layers
  across two GPUs ran without errors but produced noise after frame 0. Single-GPU offloading produces correct
  output and was also about 2× faster per step, so it's the default. With two GPUs, run two jobs in parallel
  instead (`--gpu 0` and `--gpu 1`).
- Cosmos3-Edge is a general text/image-to-video and action model. It is **not** the multi-camera driving
  simulator `Cosmos3-Nano-Sim-Auto`, which has WSM lane/cuboid controls and LiDAR. Sim-Auto is a 16B model
  (~32 GB of weights) that targets Hopper/Blackwell. Running it on 6 GB cards would need 4-bit
  quantization plus offloading, and it's the planned next step.
- Requires ~10 GB of free system RAM for the pinned transformer weights.
