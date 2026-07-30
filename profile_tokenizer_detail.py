"""Detailed micro-profiling of the tokenizer and forward pass components.

Instruments each sub-component of the forward pass separately to identify
exactly which stage dominates. Run after profile_training.py to get
fine-grained timings.

Usage:
    uv run python profile_tokenizer_detail.py experiment=pretraining/poyo_pretrain_tokenizer_sweep \
        model/tokenizer=per_channel_resample_cnn run.name=PROFILING_DETAIL run.group=DEBUGGING
"""

import logging
import os
import time
from contextlib import contextmanager

import hydra
import torch
from hydra.utils import get_class, instantiate
from omegaconf import DictConfig, OmegaConf

from foundry.config_resolvers import hydra_main_wrapper, register_resolvers
from foundry.data.datamodules.base import normalize_data_config
from foundry.seed import set_seed

os.environ.setdefault("SLURM_TMPDIR", "/tmp")
logger = logging.getLogger(__name__)


class CudaTimer:
    """CUDA-event based timer for accurate GPU profiling."""

    def __init__(self):
        self.records: dict[str, list[float]] = {}
        self._use_cuda = torch.cuda.is_available()

    @contextmanager
    def region(self, name: str):
        if self._use_cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            yield
            end.record()
            torch.cuda.synchronize()
            elapsed_ms = start.elapsed_time(end)
        else:
            t0 = time.perf_counter()
            yield
            elapsed_ms = (time.perf_counter() - t0) * 1000
        self.records.setdefault(name, []).append(elapsed_ms)

    def summary(self, title: str = "DETAILED PROFILING") -> str:
        lines = [f"\n{'=' * 80}", title, "=" * 80]
        total_all = 0
        for name, times in self.records.items():
            total = sum(times)
            avg = total / len(times)
            total_all += total
            lines.append(
                f"  {name:55s}  calls={len(times):4d}  "
                f"total={total:8.1f}ms  avg={avg:7.2f}ms"
            )
        lines.append("-" * 80)
        lines.append(f"  {'TOTAL':55s}  {' ' * 14}total={total_all:8.1f}ms")
        lines.append("=" * 80)
        return "\n".join(lines)


def _build_components(cfg):
    from foundry.data.utils import get_max_channels, get_session_configs

    normalize_data_config(cfg.data)
    dm = instantiate(cfg.data, tokenizer=None)
    dm.setup("fit")

    session_configs = get_session_configs(dm.dataset)
    OmegaConf.update(
        cfg, "hyperparameters.session_configs", session_configs, force_add=True
    )
    num_channels = get_max_channels(dm.dataset)
    OmegaConf.update(
        cfg, "hyperparameters.num_channels", num_channels, force_add=True
    )

    from foundry.tasks.config import TaskConfig

    task_configs = {}
    names = OmegaConf.to_container(cfg.task_configs, resolve=True)
    for name in names:
        path = (
            __import__("pathlib").Path(__file__).resolve().parent
            / "configs"
            / "tasks"
            / f"{name}.yaml"
        )
        tc = TaskConfig.from_yaml(path)
        task_configs[tc.name] = tc

    ModelClass = get_class(cfg.model._target_)
    model_kwargs = {
        k: instantiate(v) if OmegaConf.is_config(v) else v
        for k, v in cfg.model.items()
        if k != "_target_"
    }
    model = ModelClass(task_configs=task_configs, **model_kwargs)
    tokenizer_fn = model.tokenize if hasattr(model, "tokenize") else None
    dm.set_tokenizer(tokenizer_fn)

    return model, dm


NUM_WARMUP = 3
NUM_PROFILE = 20


@hydra.main(version_base=None, config_path="configs", config_name="config")
@hydra_main_wrapper
def main(cfg: DictConfig):
    logging.basicConfig(level=logging.INFO)
    set_seed(cfg.run.seed)
    torch.set_float32_matmul_precision("high")
    OmegaConf.resolve(cfg.run)

    model, dm = _build_components(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision = OmegaConf.select(cfg, "trainer.precision", default="32-true")
    use_amp = "bf16" in str(precision) or "16" in str(precision)
    amp_dtype = torch.bfloat16 if "bf16" in str(precision) else torch.float16

    # Initialize vocabs (normally done by VocabInitializerCallback)
    vocab_info = {}
    dataset = getattr(dm, "dataset", None)
    if dataset is None:
        dm.setup("fit")
        dataset = getattr(dm, "dataset", None)
    else:
        dm.setup("fit")
    if hasattr(dm, "get_recording_ids"):
        vocab_info["session_ids"] = dm.get_recording_ids()
    elif dataset is not None and hasattr(dataset, "get_recording_ids"):
        vocab_info["session_ids"] = dataset.get_recording_ids()
    if hasattr(dm, "get_channel_ids"):
        vocab_info["channel_ids"] = dm.get_channel_ids()
    elif dataset is not None and hasattr(dataset, "get_channel_ids"):
        vocab_info["channel_ids"] = dataset.get_channel_ids()
    model.initialize_vocabs(vocab_info)

    model = model.to(device)
    model.train()
    train_loader = dm.train_dataloader()

    timer = CudaTimer()
    step_count = 0

    # Print model info
    C_pad = model.tokenizer.channel_strategy.max_channels
    logger.info("Max channels (C_pad): %d", C_pad)
    logger.info("Sequence length: %.1fs", model.sequence_length)
    logger.info("Per-channel mode: %s", model.tokenizer.uses_per_channel)
    logger.info(
        "Has fixed token count: %s", model.tokenizer.has_fixed_token_count
    )
    logger.info("Does patching: %s", model.tokenizer.does_patching)
    if hasattr(model.tokenizer.temporal_embedding, "target_token_rate"):
        logger.info(
            "Target token rate: %.1f Hz",
            model.tokenizer.temporal_embedding.target_token_rate,
        )

    logger.info(
        "Profiling %d steps (after %d warmup)...", NUM_PROFILE, NUM_WARMUP
    )

    for batch in train_loader:
        if step_count >= NUM_WARMUP + NUM_PROFILE:
            break

        is_profiling = step_count >= NUM_WARMUP

        # Transfer to device
        batch_gpu = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                if v.dtype == torch.float64:
                    v = v.float()
                batch_gpu[k] = v.to(device, non_blocking=True)
            elif isinstance(v, dict):
                batch_gpu[k] = {
                    kk: vv.to(device, non_blocking=True)
                    if isinstance(vv, torch.Tensor)
                    else vv
                    for kk, vv in v.items()
                }
            else:
                batch_gpu[k] = v
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        target_values = batch_gpu.pop("target_values")
        _ = batch_gpu.pop("target_weights")
        batch_gpu.pop("session_id", None)
        batch_gpu.pop("absolute_start", None)
        batch_gpu.pop("eval_mask", None)

        if is_profiling:
            _profile_forward_components(
                model, batch_gpu, timer, use_amp, amp_dtype, device
            )

            # Backward on a simple forward
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                output = model(**batch_gpu, unpack_output=False)
            if output.ssl_meta:
                for tn, meta in output.ssl_meta.items():
                    target_values[tn] = meta.targets
            total_loss = torch.tensor(0.0, device=device)
            for name, preds in output.task_outputs.items():
                t = target_values.get(name)
                if t is not None and t.numel() > 0:
                    if preds.dim() == 2 and preds.shape[1] == 1:
                        preds = preds.squeeze(-1)
                    total_loss = total_loss + torch.nn.functional.mse_loss(
                        preds, t
                    )
            with timer.region("backward"):
                total_loss.backward()
        else:
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                model(**batch_gpu, unpack_output=False)

        step_count += 1

    # Print shapes from last batch for context
    for k, v in batch_gpu.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape} {v.dtype}")

    print(timer.summary("FORWARD PASS COMPONENT BREAKDOWN"))

    # Data loading benchmark
    _profile_data_loading(dm)


def _profile_forward_components(
    model, batch_gpu, timer, use_amp, amp_dtype, device
):
    """Profile individual components of the masked forward pass."""

    with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
        # 1. Channel strategy
        B, C_in, T = batch_gpu["input_values"].shape
        with timer.region("1. channel_strategy"):
            transformed = model.tokenizer.channel_strategy(
                batch_gpu["input_values"],
                input_mask=batch_gpu.get("input_mask"),
            )

        # 2. Per-channel repeat_interleave
        with timer.region("2. repeat_interleave_for_per_channel"):
            sr = batch_gpu["input_sampling_rate"].repeat_interleave(C_in)
            seq_len = batch_gpu.get("input_seq_len")
            if seq_len is not None:
                seq_len = seq_len.repeat_interleave(C_in)

        # 3. _compute_target_tokens (.item() sync)
        with timer.region("3. compute_target_tokens (.item() sync)"):
            target_tokens = (
                model.tokenizer.temporal_embedding._compute_target_tokens(
                    seq_len, sr
                )
            )

        # 4. Antialias lowpass (FFT)
        with timer.region("4. antialias_lowpass (FFT)"):
            transformed = model.tokenizer.temporal_embedding._antialias_lowpass(
                transformed, seq_len, target_tokens
            )

        # 5. Resample (grid_sample)
        with timer.region("5. resample (grid_sample)"):
            resampled = model.tokenizer.temporal_embedding._resample(
                transformed, seq_len, target_tokens
            )

        # 6. CNN
        with timer.region("6. temporal_cnn"):
            features = model.tokenizer.temporal_embedding.cnn(resampled)

        # 7. Feature projection
        with timer.region("7. feature_proj"):
            tokens = model.tokenizer.temporal_embedding.feature_proj(
                features.transpose(1, 2)
            )

        # 8. LayerNorm
        with timer.region("8. post_proj_norm"):
            tokens = model.tokenizer.post_proj_norm(tokens)

        # 9. Reassemble per-channel
        with timer.region("9. reassemble_per_channel"):
            tokens = model.tokenizer._reassemble_per_channel(
                tokens,
                B,
                batch_gpu.get("input_mask"),
                batch_gpu.get("input_channel_index"),
                model.channel_emb,
            )

        # 10. Session embedding
        with timer.region("10. session_embedding"):
            session_emb = model.session_emb(
                batch_gpu["input_session_index"]
            ).unsqueeze(1)
            tokens = tokens + session_emb

        # 11. Rotary embedding on input timestamps
        input_timestamps = batch_gpu["input_timestamps"]
        if input_timestamps.ndim == 3:
            input_timestamps = input_timestamps.reshape(B, -1)
        with timer.region("11. rotary_emb_inputs"):
            _ = model.rotary_emb(input_timestamps)

        # 12. Masking
        num_tokens = tokens.shape[1]
        C_pad = batch_gpu["input_mask"].shape[1]
        N = num_tokens // C_pad

        with timer.region("12. masking"):
            from foundry.tasks.masking import build_token_validity_mask

            token_validity = build_token_validity_mask(
                batch_gpu["input_mask"], N
            )
            mask_indices, _ = model.masking(
                num_channels=C_pad,
                num_time_tokens=N,
                channel_mask=batch_gpu["input_mask"],
                device=device,
            )

        # 13. Gather visible tokens
        with timer.region("13. gather_visible_tokens"):
            from foundry.models.masked_poyo_eeg import _compute_visible_indices

            visible_indices = _compute_visible_indices(num_tokens, mask_indices)
            D = tokens.shape[-1]
            expand_D = visible_indices.unsqueeze(-1).expand(-1, -1, D)
            visible_inputs = torch.gather(tokens, 1, expand_D)
            visible_ts = torch.gather(input_timestamps, 1, visible_indices)
            visible_ts_emb = model.rotary_emb(visible_ts)
            visible_validity = torch.gather(token_validity, 1, visible_indices)

        # 14. Build latents
        with timer.region("14. build_latents"):
            latents, latent_ts_emb = model._build_latents(
                batch_gpu["latent_index"], batch_gpu["latent_timestamps"]
            )

        # 15. Encoder cross-attention
        with timer.region("15. encoder_cross_attention"):
            latents_enc = model.backbone.encoder(
                latents,
                visible_inputs,
                latent_ts_emb,
                visible_ts_emb,
                visible_validity,
            )

        # 16. Processor self-attention
        with timer.region("16. processor_self_attention"):
            latents_proc = model.backbone.processor(latents_enc, latent_ts_emb)

        # 17. Build reconstruction queries
        masked_validity = torch.gather(token_validity, 1, mask_indices)
        with timer.region("17. build_recon_queries"):
            recon_queries, recon_ts_emb, recon_task_index = (
                model._build_reconstruction_queries(
                    session_emb,
                    mask_indices,
                    batch_gpu["input_channel_index"],
                    input_timestamps,
                    N,
                    masked_validity,
                )
            )

        # 18. Decoder cross-attention
        with timer.region("18. decoder_cross_attention"):
            output_latents = model.backbone.decoder(
                recon_queries,
                latents_proc,
                recon_ts_emb,
                latent_ts_emb,
            )

        # 19. Router
        with timer.region("19. readout_router"):
            model._route(output_latents, recon_task_index)


def _profile_data_loading(dm):
    """Measure raw DataLoader throughput."""
    loader = dm.train_dataloader()
    times = []

    for i, _batch in enumerate(loader):
        if i == 0:
            t0 = time.perf_counter()
            continue
        if i >= 51:
            break
        times.append(time.perf_counter() - t0)
        t0 = time.perf_counter()

    if times:
        avg_ms = sum(times) / len(times) * 1000
        print(f"\n{'=' * 80}")
        print("DATA LOADING THROUGHPUT")
        print("=" * 80)
        print(
            f"  Avg batch load time: {avg_ms:.1f}ms ({1000 / avg_ms:.1f} batches/s)"
        )
        print(
            f"  Min: {min(times) * 1000:.1f}ms  Max: {max(times) * 1000:.1f}ms"
        )
        print("=" * 80)


if __name__ == "__main__":
    register_resolvers()
    main()
