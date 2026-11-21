#!/usr/bin/env python
"""DROID action-policy post-training of Cosmos3-Edge on one 6 GB GPU (batch size 1).

This runs NVIDIA's ``action_policy_droid_nano`` recipe from cosmos-framework
(docs/action_policy_droid_posttrain.md): same dataset class, transforms, ``wam`` task, joint-position
actions with state, 480p concat view, losses and trainer. Only what cannot fit in 6 GB of VRAM and
30 GB of RAM changes:

| reference recipe (256 GPUs)               | here (1x RTX A2000 6 GB)                                        |
|-------------------------------------------|-----------------------------------------------------------------|
| Cosmos3-Nano base                         | Cosmos3-Edge base (4B; the recipe doc allows either)            |
| global batch 8192                         | batch 1 (``max_samples_per_batch=1``), optional grad accumulation |
| full fine-tune of the generation tower    | LoRA (rank 16) on the generation tower's attention projections; |
|                                           | the small heads the recipe trains (action in/out projections,   |
|                                           | action/time embeddings, vae<->llm projections) train in full    |
| FSDP-sharded bf16 weights on the GPUs     | frozen decoder-layer weights stay in pinned RAM as bf16 and     |
|                                           | stream to the GPU one layer at a time (forward, and again in    |
|                                           | the activation-checkpoint recompute of the backward pass)       |
| EMA copy, torch.compile, 16 loader workers | off, off, 2 workers                                            |

Usage (cosmos-framework venv; GPU via CUDA_VISIBLE_DEVICES):

    third_party/cosmos-framework/.venv/bin/python lowvram/train/make_droid_subset.py   # once
    COSMOS_TRAINING=0 third_party/cosmos-framework/.venv/bin/python \\
        -m cosmos_framework.scripts.convert_model_to_dcp \\
        --checkpoint-path checkpoints/Cosmos3-Edge -o checkpoints/Cosmos3-Edge-dcp      # once
    CUDA_VISIBLE_DEVICES=0 third_party/cosmos-framework/.venv/bin/torchrun --nproc_per_node=1 \\
        lowvram/train/train_droid_policy.py -- trainer.max_iter=200

Extra Hydra overrides after ``--`` are passed to cosmos_framework.scripts.train.
"""

from __future__ import annotations

import copy
import functools
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FRAMEWORK = ROOT / "third_party" / "cosmos-framework"
TOML = Path(__file__).resolve().parent / "action_policy_droid_edge_lowvram.toml"

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("LOWVRAM_STREAM_DTYPE", "bf16")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("DROID_ROOT", str(ROOT / "checkpoints" / "droid_subset" / "droid_plus_lerobot_640x360_20260412"))
os.environ.setdefault("BASE_CHECKPOINT_PATH", str(ROOT / "checkpoints" / "Cosmos3-Edge-dcp"))
os.environ.setdefault("BASE_HF_CHECKPOINT", str(ROOT / "checkpoints" / "Cosmos3-Edge"))
os.environ.setdefault("IMAGINAIRE_OUTPUT_ROOT", str(ROOT / "outputs" / "train"))
if "WAN_VAE_PATH" not in os.environ:
    found = sorted(Path.home().glob(".cache/huggingface/hub/models--Wan-AI--Wan2.2-TI2V-5B/snapshots/*/Wan2.2_VAE.pth"))
    if found:
        os.environ["WAN_VAE_PATH"] = str(found[0])

sys.path.insert(0, str(ROOT / "lowvram" / "simauto"))
sys.path.insert(0, str(FRAMEWORK))


def _preload_npp() -> None:
    """torchcodec's CUDA build links libnpp*.so.13, which the venv ships under nvidia/cu13/lib
    but not on the loader path; load them globally so the dataloader can decode DROID videos."""
    import ctypes
    import site

    for base in site.getsitepackages():
        lib = Path(base) / "nvidia" / "cu13" / "lib"
        names = sorted(lib.glob("libnpp*.so.*"), key=lambda p: not p.name.startswith("libnppc."))
        for path in names:
            ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)


_preload_npp()

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import int8_stream  # noqa: E402

# Small modules the reference recipe trains (its optimizer keys_to_select, minus the
# generation tower ``moe_gen``, which gets LoRA instead).
TRAIN_FULLY = ("time_embedder", "vae2llm", "llm2vae", "action2llm", "llm2action", "action_modality_embed")


class StreamedLoraLinear(int8_stream.StreamedInt8Linear):
    """A LoRA-injected linear whose frozen base weight streams from pinned RAM.

    Same math as ``cosmos_framework.utils.generator.lora.LoraInjectedLinear``; the adapters are
    ordinary trainable modules that stay on the GPU.
    """

    def __init__(self, lora_module: nn.Module, name: str) -> None:
        super().__init__(lora_module.in_features, lora_module.out_features, name)
        self.lora_A = lora_module.lora_A
        self.lora_B = lora_module.lora_B
        self.lora_dropout = lora_module.lora_dropout
        self.lora_scale = float(lora_module._lora_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = super().forward(x)
        adapter_input = self.lora_dropout(x).to(self.lora_A.weight.dtype)
        return base + self.lora_scale * self.lora_B(self.lora_A(adapter_input)).to(base.dtype)


def _swap_to_streamed(net: nn.Module) -> dict[str, int]:
    """Replace every decoder-layer linear with a streamed one (keeping LoRA adapters)."""
    from cosmos_framework.utils.generator.lora import LoraInjectedLinear

    counts = {"lora": 0, "plain": 0, "embedding": 0, "lm_head": 0}
    order = []
    for name, module in list(net.named_modules()):
        if isinstance(module, nn.Embedding) and name.endswith("embed_tokens"):
            # Frozen 131k x 2048 token table: lookups on the CPU instead of 0.5 GB of VRAM.
            int8_stream._replace(
                net, name, int8_stream.CpuEmbedding(module.num_embeddings, module.embedding_dim, module.padding_idx, name)
            )
            counts["embedding"] += 1
            continue
        if isinstance(module, nn.Linear) and name.endswith("lm_head"):  # text logits; unused by this loss
            int8_stream._replace(net, name, int8_stream.StreamedInt8Linear(module.in_features, module.out_features, name))
            counts["lm_head"] += 1
            continue
        if ".layers." not in name or not isinstance(module, nn.Linear) or ".lora_" in name:
            continue
        if module.bias is not None:
            raise NotImplementedError(f"{name}: biased linears are not streamed")
        if isinstance(module, LoraInjectedLinear):
            new = StreamedLoraLinear(module, name)
            counts["lora"] += 1
        else:
            new = int8_stream.StreamedInt8Linear(module.in_features, module.out_features, name)
            counts["plain"] += 1
        int8_stream._replace(net, name, new)
        order.append(new)
    for i, m in enumerate(order):
        m.index = i
    int8_stream.PREFETCH.order = order
    return counts


# Read-only ``.weight`` for streamed linears (the pinned CPU copy). Framework diagnostics such as the
# MoE stability/specialization callbacks read ``layers[0].self_attn.q_proj.weight`` to find the
# FSDP device mesh; on this unsharded model they then report nothing, as for any single-GPU run.
int8_stream.StreamedInt8Linear.weight = property(lambda self: self.q_cpu)


def install_patches() -> None:
    from cosmos_framework.inference import model as inference_model
    from cosmos_framework.model.generator import omni_mot_model
    from cosmos_framework.utils.generator import lora

    # 1) Stream the frozen decoder weights. LoRA is injected before parallelize, so the swap
    #    sees LoraInjectedLinear modules; the base weights come straight from the HF checkpoint.
    original_parallelize = omni_mot_model.parallelize_vfm_network

    @functools.wraps(original_parallelize)
    def parallelize_then_stream(net, *args, **kwargs):
        counts = _swap_to_streamed(net)
        print(f"[lowvram] streamed decoder linears: {counts}", flush=True)
        int8_stream.load_streamed_weights(
            net, Path(os.environ["BASE_HF_CHECKPOINT"]), inference_model._diffusers_to_net_key, torch.device("cpu")
        )
        return original_parallelize(net, *args, **kwargs)

    omni_mot_model.parallelize_vfm_network = parallelize_then_stream

    # 2) LoRA adapters of the streamed modules are initialized like LoraInjectedLinear's, and
    #    kept in fp32 so AdamW updates them precisely while the rest of the model runs in bf16.
    original_init = lora.init_lora_weights_post_materialization

    @torch.no_grad()
    def init_lora(network: nn.Module) -> None:
        original_init(network)
        for module in network.modules():
            if isinstance(module, StreamedLoraLinear):
                module.lora_A.float()
                module.lora_B.float()
                nn.init.kaiming_uniform_(module.lora_A.weight, a=5**0.5)
                nn.init.zeros_(module.lora_B.weight)

    lora.init_lora_weights_post_materialization = init_lora

    # 3) Besides the adapters, train the small heads the recipe trains (incl. the fresh action heads).
    original_scope = lora.apply_lora_trainable_scope

    @functools.wraps(original_scope)
    def scope_with_heads(network: nn.Module, **kwargs):
        summary = original_scope(network, **kwargs)
        extra = 0
        for name, parameter in network.named_parameters():
            if any(key in name for key in TRAIN_FULLY):
                if not parameter.requires_grad:
                    extra += parameter.numel()
                parameter.requires_grad_(True)
        print(f"[lowvram] also training {extra:,} head parameters: {', '.join(TRAIN_FULLY)}", flush=True)
        return summary

    lora.apply_lora_trainable_scope = scope_with_heads


def register_experiment() -> None:
    """``action_policy_droid_edge_lowvram``: the DROID recipe on the Edge base, sized for 6 GB."""
    from hydra.core.config_store import ConfigStore

    from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_droid_nano import (
        action_policy_droid_nano,
    )
    from cosmos_framework.configs.base.experiment.sft.models.edge_model_config import EDGE_MODEL_CONFIG

    exp = copy.deepcopy(action_policy_droid_nano)
    model = copy.deepcopy(EDGE_MODEL_CONFIG)
    # Recipe deltas applied to the base model config (see action_policy_droid_nano.py).
    model["tokenizer"]["encode_exact_durations"] = [33]
    # VAE-encode 4 frames per chunk instead of 24 (same latents, a fraction of the activations).
    model["tokenizer"]["encode_chunk_frames"] = {k: 4 for k in model["tokenizer"]["encode_chunk_frames"]}
    model["max_num_tokens_after_packing"] = -1
    model["rectified_flow_training_config"]["loss_scale"] = 10.0
    # Low-VRAM deltas.
    model["lora_enabled"] = True
    model["lora_rank"] = 16
    model["lora_alpha"] = 32
    model["ema"]["enabled"] = False
    model["compile"]["enabled"] = False
    model["parallelism"]["data_parallel_shard_degree"] = 1
    model["parallelism"]["fsdp_master_dtype"] = "bfloat16"
    model["activation_checkpointing"] = dict(mode="full")
    exp["model"]["config"] = model

    loader = exp["dataloader_train"]
    loader["max_samples_per_batch"] = 1
    inner = loader["dataloader"]
    inner["batch_size"] = 1
    inner["num_workers"] = 2
    inner["prefetch_factor"] = 2
    inner["persistent_workers"] = True
    # This cosmos-framework revision's get_action_droid_sft_dataset() has no
    # filter_dict_source_prefix parameter (the recipe sets it to "", i.e. unused).
    inner["datasets"]["droid"]["dataset"].pop("filter_dict_source_prefix", None)
    exp["job"]["name"] = "action_policy_droid_edge_lowvram"
    exp["trainer"]["callbacks"]["manual_gc"]["every_n"] = 1
    ConfigStore.instance().store(group="experiment", package="_global_", name="action_policy_droid_edge_lowvram", node=exp)


def main() -> None:
    os.chdir(FRAMEWORK)  # model configs reference repo-relative files, as in NVIDIA's launchers
    install_patches()
    register_experiment()
    from cosmos_framework.scripts import train

    extra = sys.argv[1:]
    if extra and extra[0] == "--":
        extra = extra[1:]
    sys.argv = [sys.argv[0], "--sft-toml", str(TOML), "--", *extra]
    raise SystemExit(train.main())


if __name__ == "__main__":
    main()
