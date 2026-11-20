# Cosmos-Edge-Sim

Run NVIDIA Cosmos 3 world models on small Ampere GPUs. This was tested on a desktop with **2× RTX A2000 (6 GB each, sm_86) and 30 GB of RAM**.

| Model | What it does | Script | GPUs |
|---|---|---|---|
| [Cosmos3-Nano-Sim-Auto](https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Auto) (16B) | Closed-loop driving simulator: camera video that follows WSM lane/cuboid controls, optional LiDAR | `lowvram/simauto/run_sim_auto.py` | 2 |
| [Cosmos3-Edge](https://huggingface.co/nvidia/Cosmos3-Edge) (4B) | General image/text-to-video | `lowvram/run_edge.py` | 1 |

---

## Cosmos3-Nano-Sim-Auto on 2× 6 GB GPUs

NVIDIA's runtime ([cosmos-framework](https://github.com/NVIDIA/cosmos-framework)) assumes Hopper or Blackwell GPUs with ≥ 80 GB of memory. It puts all ~30 GB of bf16 weights on every GPU and needs FlashAttention-4. `lowvram/simauto/run_sim_auto.py` runs the unmodified framework with these patches:

| Patch | File |
|---|---|
| FlashAttention-4 "gapped replay" attention is replaced by a PyTorch varlen attention with the same contract. It is unit-tested against SDPA. | `fa_shim.py` |
| 505 MoT decoder linears are int8 per-channel weights (12.9 GB) held in pinned CPU RAM and streamed layer by layer to GPU 0, with the next layer prefetched. The token embedding stays on CPU and the unused text `lm_head` is dropped. Only ~0.6 GB stays resident on GPU 0. | `int8_stream.py` |
| The Wan2.2 video VAE runs on GPU 1. It is loaded on CPU first, encodes 4-frame chunks, and decodes in two overlapping width strips; the inactive strip's cache is parked on GPU 0. | `run_sim_auto.py`, `vae_tiling.py` |
| The LiDAR VAE runs on GPU 0. | `run_sim_auto.py` |
| All GPU↔GPU copies are staged through host RAM. See "Hardware note" below. | `run_sim_auto.py` |

### Result

| Request | Load | Generation | Total |
|---|---|---|---|
| 1 camera (`camera_front_wide_120fov`), 49 frames, 480p, RGB, first-frame conditioned | ~1 min | 3 min 10 s (~1.6 s per frame) | 263 s |

The output follows the daytime sample's real scene and its WSM controls: road layout, oncoming cars, roadside fence and gateposts.

### Setup

```bash
git clone https://github.com/NVIDIA/cosmos-framework third_party/cosmos-framework
cd third_party/cosmos-framework
curl -LsSf https://astral.sh/uv/0.11.29/install.sh | env UV_INSTALL_DIR="$HOME/.local/bin" sh
~/.local/bin/uv sync --locked --python 3.13 --group cu130-train --extra train --extra guardrail
cd ../..
hf download nvidia/Cosmos3-Nano-Sim-Auto --local-dir checkpoints/Cosmos3-Nano-Sim-Auto   # ~32 GB
```

The Wan2.2 VAE (`Wan-AI/Wan2.2-TI2V-5B`) downloads automatically on the first run.

### Usage

```bash
PY=third_party/cosmos-framework/.venv/bin/python
# Build a reduced request from NVIDIA's published daytime sample (fetched on first use)
$PY lowvram/simauto/prepare_request.py --views camera_front_wide_120fov --num-frames 49 \
    -o outputs/sim_auto/requests/front_49f.json
# Generate. Extra arguments go straight to cosmos_framework.scripts.inference
$PY lowvram/simauto/run_sim_auto.py -i outputs/sim_auto/requests/front_49f.json \
    -o outputs/sim_auto/front_49f --checkpoint-path checkpoints/Cosmos3-Nano-Sim-Auto
```

Videos are written to `outputs/sim_auto/front_49f/daytime_scene/vision_view00_*.mp4`.

Environment knobs:

| Variable | Default | Meaning |
|---|---|---|
| `SIM_AUTO_VAE_GPU` | 1 | GPU for the video VAE |
| `SIM_AUTO_LIDAR_GPU` | 0 | GPU for the LiDAR VAE |
| `SIM_AUTO_VAE_ENCODE_FRAMES` | 4 | Frames per VAE encode chunk |
| `SIM_AUTO_VAE_TILE_LATENT` | 30 | Latent columns per decode strip |
| `SIM_AUTO_VAE_TILE_OVERLAP` | 8 | Overlap between decode strips, in latent columns |
| `SIM_AUTO_VAE_CACHE_DEVICE` | `cuda:0` | Where the inactive strip's decoder cache waits |
| `SIM_AUTO_DEBUG` | — | Set to 1 to print per-layer activation and VAE input/output statistics |

### Limits

- Tested only with one camera at 480p, RGB, 49 frames. More cameras, longer rollouts and joint LiDAR are untested; the rolling KV cache and activations grow with the number of views.
- The int8 weights add some quantization error compared with NVIDIA's bf16 reference. The attention shim is slower than FA4.
- About 13 GB of RAM stays locked for the weights. Leave about 6 GB free beyond that.
- Run it outside Claude Code's background-task memory guard (for example `setsid nohup ... &`) or in your own terminal.

### Hardware note: GPU peer-to-peer copies are broken on this machine

CUDA reports peer access between the two A2000s, and `nvidia-smi topo -p2p r` shows "OK". But a direct `tensor.to("cuda:1")` from `cuda:0` silently returns **all zeros**; staging through host RAM works. This is the usual symptom of the IOMMU or PCIe ACS blocking P2P.

- Everything here stages GPU↔GPU copies through the CPU.
- Disabling the IOMMU or ACS in the BIOS or kernel options would let direct copies work.
- The same bug is why splitting Cosmos3-Edge's layers across both GPUs with `accelerate` produced noise.

Quick check:

```bash
python -c "import torch;a=torch.ones(10,device='cuda:0');print(a.to('cuda:1'))"   # zeros = broken P2P
```

---

## Cosmos3-Edge on one 6 GB GPU

The official recipe (`pipe.to("cuda")`) needs about 10 GB of VRAM for the bf16 weights alone. `lowvram/run_edge.py` brings the peak down to **about 3 GB**:

| Technique | What it saves |
|---|---|
| Diffusers group offloading: transformer weights stay in pinned CPU RAM, 2 decoder layers at a time are streamed to the GPU on a side CUDA stream | ~6.7 GB of transformer weights |
| VAE (Wan2.2) kept on CPU; moved to the GPU only for the conditioning encode and the final tiled decode | ~1.4 GB during denoising |
| Image-to-video encodes only the conditioning frame (the stock pipeline VAE-encodes the image repeated over all frames, but keeps only latent frame 0) | ~2.5 GB peak at 480p |
| `cosmos_guardrail` safety checker off by default (`--safety` turns it on) | several GB |

| Frames | Resolution | Steps | Denoise | VAE decode | Total | Peak VRAM |
|---|---|---|---|---|---|---|
| 33 | 832×480 | 20 | 92 s | 26 s | 132 s | 2.2 GB |
| 121 (model-card default) | 832×480 | 20 | 217 s | 94 s | 321 s | 3.0 GB |

```bash
scripts/setup_env.sh            # conda env "cosmosim": Python 3.12, torch 2.11 cu128, diffusers 0.41.0
conda activate cosmosim
scripts/download_edge.sh        # ~8.6 GB into checkpoints/Cosmos3-Edge
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python lowvram/run_edge.py --gpu 0
python lowvram/run_edge.py --gpu 1 --image my_frame.jpg --prompt "Dashcam view, light rain, ..." \
    --num-frames 61 -o outputs/my_run.mp4
```

Flags:
- `--offload-blocks N`: layers streamed per group. Higher is faster but uses more VRAM.
- `--num-frames`, `--steps`, `--guidance`, `--flow-shift`, `--seed`.

With two GPUs, run two Edge jobs in parallel, one with `--gpu 0` and one with `--gpu 1`.
