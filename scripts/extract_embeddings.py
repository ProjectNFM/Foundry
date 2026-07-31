"""Extract backbone and channel embeddings for visualization and analysis.

Reuses Hydra experiment configs to build the same model and data pipeline,
loads a checkpoint (or uses random init), then extracts latent representations
and optionally dynamic channel embeddings from the validation set.

The script produces numpy arrays and metadata JSON files that can be consumed
by downstream analysis scripts (e.g., t-SNE/PCA visualization).

Extraction targets:
    - **backbone**: Pool processed latent tokens → (N, embed_dim) per sample.
    - **channel_emb**: Dynamic channel embeddings → (N*C, ch_emb_dim) per
      channel per sample (only when ``extract.extract_channel_emb=true`` and
      the model uses ``channel_emb_mode=dynamic``).

Usage:
    # Backbone embeddings (default):
    uv run python scripts/extract_embeddings.py \\
        experiment=sleep_staging/poyo_kemp_allsess \\
        run.pretrained_checkpoint=/path/to/checkpoint.ckpt \\
        extract.output_dir=outputs/embeddings/my_run

    # Backbone + dynamic channel embeddings:
    uv run python scripts/extract_embeddings.py \\
        experiment=sleep_staging/poyo_kemp_allsess \\
        model/tokenizer=per_channel_resample_cnn \\
        model.channel_emb_mode=dynamic \\
        model/session_emb=disabled \\
        run.pretrained_checkpoint=/path/to/checkpoint.ckpt \\
        run.pretrained_transfer_mode=permissive \\
        extract.output_dir=outputs/embeddings/my_run \\
        extract.extract_channel_emb=true
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.utils import get_class, instantiate
from omegaconf import DictConfig, OmegaConf
from rich.logging import RichHandler
from torch.utils.data import DataLoader
from tqdm import tqdm

from foundry.config_resolvers import register_resolvers
from foundry.data.datamodules.base import normalize_data_config
from foundry.seed import set_seed
from foundry.training.pretrained import TransferMode, load_pretrained_weights

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def setup_logging():
    logging.basicConfig(
        level="INFO",
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(rich_tracebacks=True, markup=True, show_path=False)
        ],
        force=True,
    )


# ---------------------------------------------------------------------------
# Model & data construction (mirrors main.py logic)
# ---------------------------------------------------------------------------


def _build_model_and_data(cfg: DictConfig):
    """Build model and datamodule from config."""
    from foundry.data.utils import get_max_channels, get_session_configs
    from foundry.tasks.config import TaskConfig

    _TASKS_DIR = Path(__file__).resolve().parent.parent / "configs" / "tasks"

    names = OmegaConf.to_container(cfg.task_configs, resolve=True)
    task_configs = {}
    for name in names:
        path = _TASKS_DIR / f"{name}.yaml"
        tc = TaskConfig.from_yaml(path)
        task_configs[tc.name] = tc

    normalize_data_config(cfg.data)
    datamodule = instantiate(cfg.data, tokenizer=None)
    datamodule._task_configs = task_configs
    datamodule.setup("fit")

    session_configs = OmegaConf.select(
        cfg, "hyperparameters.session_configs", default=None
    )
    if session_configs is None:
        session_configs = get_session_configs(datamodule.dataset)
        OmegaConf.update(
            cfg,
            "hyperparameters.session_configs",
            session_configs,
            force_add=True,
        )

    num_channels = OmegaConf.select(
        cfg, "hyperparameters.num_channels", default=None
    )
    if num_channels is None:
        num_channels = get_max_channels(datamodule.dataset)
        OmegaConf.update(
            cfg, "hyperparameters.num_channels", num_channels, force_add=True
        )

    ModelClass = get_class(cfg.model._target_)
    model_kwargs = {
        k: instantiate(v) if OmegaConf.is_config(v) else v
        for k, v in cfg.model.items()
        if k != "_target_"
    }

    session_emb_cfg = model_kwargs.pop("session_emb", None)
    if session_emb_cfg is not None:
        if OmegaConf.is_config(session_emb_cfg):
            session_emb_cfg = OmegaConf.to_container(
                session_emb_cfg, resolve=True
            )
        session_emb_cfg.pop("session_context", None)
        model_kwargs.update(session_emb_cfg)

    model = ModelClass(task_configs=task_configs, **model_kwargs)

    tokenizer = model.tokenize if hasattr(model, "tokenize") else None
    datamodule.set_tokenizer(tokenizer)

    return model, datamodule


def _initialize_vocabs(model, datamodule):
    """Initialize lazy vocabularies from the datamodule."""
    if not hasattr(model, "has_lazy_vocabs") or not model.has_lazy_vocabs():
        return

    vocab_info = {}
    if hasattr(datamodule, "get_recording_ids"):
        vocab_info["session_ids"] = datamodule.get_recording_ids()
    if hasattr(datamodule, "get_channel_ids"):
        vocab_info["channel_ids"] = datamodule.get_channel_ids()
    model.initialize_vocabs(vocab_info)


def _load_checkpoint_if_needed(model, cfg: DictConfig):
    """Load pretrained weights if a checkpoint is specified."""
    pretrained_ckpt = OmegaConf.select(
        cfg, "run.pretrained_checkpoint", default=None
    )
    if pretrained_ckpt:
        transfer_mode_str = OmegaConf.select(
            cfg, "run.pretrained_transfer_mode", default="strict"
        )
        transfer_mode = TransferMode(transfer_mode_str)
        report = load_pretrained_weights(
            model, pretrained_ckpt, freeze=False, mode=transfer_mode
        )
        logger.info(
            "Loaded pretrained weights: %d parameters", len(report.loaded)
        )
    else:
        logger.info("No checkpoint specified — using random initialization.")


# ---------------------------------------------------------------------------
# Hook-based extractors
# ---------------------------------------------------------------------------


class LatentExtractor:
    """Hook-based extractor that captures processed latent representations."""

    def __init__(self, model):
        self.model = model
        self.latents = None
        self._hook = None

    def _hook_fn(self, module, input, output):
        self.latents = output.detach()

    def register(self):
        self._hook = self.model.backbone.processor.register_forward_hook(
            self._hook_fn
        )

    def remove(self):
        if self._hook is not None:
            self._hook.remove()


class ChannelEmbeddingExtractor:
    """Hook-based extractor that captures dynamic channel embeddings.

    Hooks into ``model.relative_channel_encoder`` to capture the
    (B, C, channel_emb_dim) output before it is fused into input tokens.
    """

    def __init__(self, model):
        self.model = model
        self.channel_embs = None
        self._hook = None

    def _hook_fn(self, module, input, output):
        self.channel_embs = output.detach()

    def register(self):
        if self.model.relative_channel_encoder is None:
            raise ValueError(
                "Cannot extract channel embeddings: model does not have "
                "a RelativeChannelEncoder (channel_emb_mode != 'dynamic')."
            )
        self._hook = self.model.relative_channel_encoder.register_forward_hook(
            self._hook_fn
        )

    def remove(self):
        if self._hook is not None:
            self._hook.remove()


# ---------------------------------------------------------------------------
# Extraction result container
# ---------------------------------------------------------------------------


@dataclass
class ExtractionResult:
    """Container for all extracted data from a validation pass."""

    embeddings: np.ndarray  # (N, embed_dim) backbone embeddings
    labels: np.ndarray  # (N,) integer class labels
    session_ids: list[str]  # (N,) session identifier per sample

    channel_embs: np.ndarray | None = None  # (M, ch_emb_dim)
    channel_names: list[str] = field(default_factory=list)  # (M,)
    channel_session_ids: list[str] = field(default_factory=list)  # (M,)
    channel_labels: np.ndarray | None = None  # (M,)


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------


def _build_channel_token_map(model) -> dict[int, str]:
    """Build reverse mapping from channel token indices to channel names."""
    if hasattr(model, "channel_emb") and hasattr(model.channel_emb, "vocab"):
        return {v: k for k, v in model.channel_emb.vocab.items()}
    return {}


@torch.no_grad()
def extract_all(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
    extract_channel_emb: bool = False,
) -> ExtractionResult:
    """Extract backbone embeddings and optionally dynamic channel embeddings.

    Args:
        model: The POYO EEG model.
        dataloader: Validation dataloader.
        device: Torch device.
        max_batches: Cap on number of batches to process.
        extract_channel_emb: If True, also extract per-channel embeddings
            from the RelativeChannelEncoder.

    Returns:
        ExtractionResult with all extracted arrays and metadata.
    """
    model.eval()
    model.to(device)

    latent_extractor = LatentExtractor(model)
    latent_extractor.register()

    ch_extractor = None
    if extract_channel_emb:
        ch_extractor = ChannelEmbeddingExtractor(model)
        ch_extractor.register()

    token_map = _build_channel_token_map(model)

    all_embeddings = []
    all_labels = []
    all_session_ids = []

    all_channel_embs = []
    all_channel_names = []
    all_channel_session_ids = []
    all_channel_labels = []

    num_batches = 0
    for batch in tqdm(dataloader, desc="Extracting embeddings"):
        if max_batches is not None and num_batches >= max_batches:
            break

        target_values = batch.pop("target_values")
        batch.pop("target_weights", None)
        session_ids_batch = batch.pop("session_id", None)
        batch.pop("absolute_start", None)
        batch.pop("eval_mask", None)

        input_channel_index = batch.get("input_channel_index")
        input_mask = batch.get("input_mask")

        batch_device = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        model(**batch_device, unpack_output=False)

        # --- backbone embeddings ---
        latents = latent_extractor.latents
        pooled = latents.mean(dim=1)  # (B, embed_dim)
        all_embeddings.append(pooled.cpu().numpy())

        # --- labels ---
        labels = None
        for task_name, targets in target_values.items():
            if "sleep" in task_name:
                labels = targets.numpy()
                break
        if labels is None:
            labels = list(target_values.values())[0].numpy()
        all_labels.append(labels)

        # --- session IDs ---
        if session_ids_batch is not None:
            if isinstance(session_ids_batch, (list, tuple)):
                all_session_ids.extend([str(s) for s in session_ids_batch])
            else:
                all_session_ids.extend([str(s) for s in session_ids_batch])

        # --- channel embeddings ---
        if ch_extractor is not None and ch_extractor.channel_embs is not None:
            ch_emb = ch_extractor.channel_embs.cpu()  # (B, C, D_ch)
            B, C, D_ch = ch_emb.shape

            if input_mask is not None:
                mask = input_mask.bool()  # (B, C)
            else:
                mask = torch.ones(B, C, dtype=torch.bool)

            for b in range(B):
                valid_channels = mask[b]  # (C,)
                valid_embs = ch_emb[b, valid_channels]  # (n_valid, D_ch)
                all_channel_embs.append(valid_embs.numpy())

                if input_channel_index is not None:
                    ch_tokens = input_channel_index[b, valid_channels].tolist()
                    ch_names = [token_map.get(t, f"ch_{t}") for t in ch_tokens]
                else:
                    ch_names = [f"ch_{i}" for i in range(valid_embs.shape[0])]
                all_channel_names.extend(ch_names)

                sid = (
                    str(session_ids_batch[b])
                    if session_ids_batch is not None
                    else "unknown"
                )
                all_channel_session_ids.extend([sid] * len(ch_names))
                all_channel_labels.extend([int(labels[b])] * len(ch_names))

        num_batches += 1

    latent_extractor.remove()
    if ch_extractor is not None:
        ch_extractor.remove()

    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    valid_mask = labels >= 0
    embeddings = embeddings[valid_mask]
    labels = labels[valid_mask]
    if all_session_ids:
        all_session_ids = [s for s, v in zip(all_session_ids, valid_mask) if v]

    result = ExtractionResult(
        embeddings=embeddings,
        labels=labels,
        session_ids=all_session_ids,
    )

    if all_channel_embs:
        ch_embs_arr = np.concatenate(all_channel_embs, axis=0)
        ch_labels_arr = np.array(all_channel_labels)

        ch_valid = ch_labels_arr >= 0
        result.channel_embs = ch_embs_arr[ch_valid]
        result.channel_names = [
            n for n, v in zip(all_channel_names, ch_valid) if v
        ]
        result.channel_session_ids = [
            s for s, v in zip(all_channel_session_ids, ch_valid) if v
        ]
        result.channel_labels = ch_labels_arr[ch_valid]

    return result


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------


def save_extraction(result: ExtractionResult, output_dir: Path):
    """Save extraction results to disk as numpy arrays and JSON metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "embeddings.npy", result.embeddings)
    np.save(output_dir / "labels.npy", result.labels)

    metadata = {
        "n_samples": int(len(result.embeddings)),
        "embed_dim": int(result.embeddings.shape[1]),
        "n_sessions": len(set(result.session_ids)),
        "session_ids": list(set(result.session_ids)),
    }

    if result.session_ids:
        with open(output_dir / "session_ids.json", "w") as f:
            json.dump(result.session_ids, f)

    if result.channel_embs is not None:
        np.save(output_dir / "channel_embs.npy", result.channel_embs)
        np.save(output_dir / "channel_labels.npy", result.channel_labels)
        with open(output_dir / "channel_names.json", "w") as f:
            json.dump(result.channel_names, f)
        with open(output_dir / "channel_session_ids.json", "w") as f:
            json.dump(result.channel_session_ids, f)

        metadata["n_channel_embs"] = int(len(result.channel_embs))
        metadata["channel_emb_dim"] = int(result.channel_embs.shape[1])
        metadata["unique_channels"] = sorted(
            set(_short_channel_name(n) for n in result.channel_names)
        )

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Saved extraction results to %s", output_dir)


def _short_channel_name(full_name: str) -> str:
    """Extract short channel name from session-scoped ID.

    E.g. 'SC4001E0-PSG/EEG Fpz-Cz' → 'EEG Fpz-Cz'
    """
    if "/" in full_name:
        return full_name.split("/", 1)[1]
    return full_name


# ---------------------------------------------------------------------------
# Backward-compatible wrapper (keeps old API working)
# ---------------------------------------------------------------------------


@torch.no_grad()
def extract_embeddings(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract backbone latent embeddings and labels (legacy API).

    Returns:
        (embeddings, labels) where embeddings is (N, embed_dim) and
        labels is (N,) with integer sleep stage classes.
    """
    result = extract_all(model, dataloader, device, max_batches)
    return result.embeddings, result.labels


# ---------------------------------------------------------------------------
# Visualization (kept for backward compatibility with exp 008)
# ---------------------------------------------------------------------------


def compute_visualizations(
    embeddings: np.ndarray,
    labels: np.ndarray,
    output_dir: Path,
    class_names: list[str] | None = None,
):
    """Compute t-SNE, PCA, and silhouette scores; save results."""
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score

    output_dir.mkdir(parents=True, exist_ok=True)

    if class_names is None:
        class_names = ["W", "N1", "N2", "N3", "REM"]

    logger.info("Computing PCA (n_components=50 then 2)...")
    pca_50 = PCA(n_components=min(50, embeddings.shape[1]))
    emb_pca50 = pca_50.fit_transform(embeddings)

    pca_2 = PCA(n_components=2)
    emb_pca2 = pca_2.fit_transform(embeddings)

    logger.info("Computing t-SNE...")
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        max_iter=1000,
        random_state=42,
        init="pca",
    )
    emb_tsne = tsne.fit_transform(emb_pca50)

    sil_score = silhouette_score(
        emb_pca50[:5000] if len(emb_pca50) > 5000 else emb_pca50,
        labels[:5000] if len(labels) > 5000 else labels,
    )
    logger.info("Silhouette score: %.4f", sil_score)

    np.save(output_dir / "embeddings.npy", embeddings)
    np.save(output_dir / "labels.npy", labels)
    np.save(output_dir / "tsne_2d.npy", emb_tsne)
    np.save(output_dir / "pca_2d.npy", emb_pca2)

    metadata = {
        "n_samples": int(len(embeddings)),
        "embed_dim": int(embeddings.shape[1]),
        "silhouette_score": float(sil_score),
        "pca_explained_variance_ratio": pca_50.explained_variance_ratio_.tolist(),
        "class_names": class_names,
        "label_counts": {
            class_names[i]: int((labels == i).sum())
            for i in range(len(class_names))
            if i < labels.max() + 1
        },
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    colors = plt.cm.Set1(np.linspace(0, 1, len(class_names)))

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    for i, name in enumerate(class_names):
        mask = labels == i
        if mask.sum() == 0:
            continue
        ax.scatter(
            emb_tsne[mask, 0],
            emb_tsne[mask, 1],
            c=[colors[i]],
            label=name,
            alpha=0.4,
            s=8,
        )
    ax.legend(markerscale=3)
    ax.set_title(f"t-SNE (silhouette={sil_score:.3f})")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    fig.tight_layout()
    fig.savefig(output_dir / "tsne_by_stage.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    for i, name in enumerate(class_names):
        mask = labels == i
        if mask.sum() == 0:
            continue
        ax.scatter(
            emb_pca2[mask, 0],
            emb_pca2[mask, 1],
            c=[colors[i]],
            label=name,
            alpha=0.4,
            s=8,
        )
    ax.legend(markerscale=3)
    ax.set_title("PCA (first 2 components)")
    ax.set_xlabel(f"PC1 ({pca_50.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca_50.explained_variance_ratio_[1]:.1%})")
    fig.tight_layout()
    fig.savefig(output_dir / "pca_by_stage.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    cumvar = np.cumsum(pca_50.explained_variance_ratio_)
    ax.plot(range(1, len(cumvar) + 1), cumvar, "o-")
    ax.axhline(0.95, ls="--", color="gray", label="95% variance")
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_title("PCA explained variance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "pca_variance.png", dpi=150)
    plt.close(fig)

    logger.info("Saved outputs to %s", output_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    setup_logging()
    set_seed(cfg.run.seed)

    extract_cfg = OmegaConf.select(cfg, "extract", default=OmegaConf.create({}))
    output_dir = Path(
        OmegaConf.select(
            extract_cfg, "output_dir", default="outputs/embeddings/default"
        )
    )
    max_batches = OmegaConf.select(extract_cfg, "max_batches", default=None)
    do_channel_emb = OmegaConf.select(
        extract_cfg, "extract_channel_emb", default=False
    )

    OmegaConf.resolve(cfg.run)

    logger.info("Building model and data...")
    model, datamodule = _build_model_and_data(cfg)

    _initialize_vocabs(model, datamodule)
    _load_checkpoint_if_needed(model, cfg)

    val_loader = datamodule.val_dataloader()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(
        "Extracting embeddings (device=%s, max_batches=%s, channel_emb=%s)...",
        device,
        max_batches,
        do_channel_emb,
    )

    result = extract_all(
        model,
        val_loader,
        device,
        max_batches=max_batches,
        extract_channel_emb=do_channel_emb,
    )
    logger.info("Extracted %d samples with dim=%d", *result.embeddings.shape)
    if result.channel_embs is not None:
        logger.info(
            "Extracted %d channel embeddings with dim=%d",
            *result.channel_embs.shape,
        )

    save_extraction(result, output_dir)

    task_configs = model.task_configs
    class_names = None
    for cfg_entry in task_configs.values():
        if cfg_entry.class_mapping is not None:
            class_names = cfg_entry.get_class_names()
            break

    compute_visualizations(
        result.embeddings, result.labels, output_dir, class_names
    )


if __name__ == "__main__":
    register_resolvers()
    main()
