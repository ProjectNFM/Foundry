"""Extract backbone embeddings for t-SNE/PCA visualization.

Reuses Hydra experiment configs to build the same model and data pipeline,
loads a checkpoint (or uses random init), then extracts latent representations
from the backbone for the validation set.

Usage:
    # Pretrained CWT-CNN:
    uv run python scripts/extract_embeddings.py \
        experiment=sleep_staging/poyo_kemp_linear_probe \
        model/tokenizer=per_channel_cwt_cnn \
        run.init_mode=pretrained \
        extract.output_dir=outputs/embeddings/008_pretrained_cwt_cnn

    # Random-init CWT-CNN:
    uv run python scripts/extract_embeddings.py \
        experiment=sleep_staging/poyo_kemp_linear_probe \
        model/tokenizer=per_channel_cwt_cnn \
        run.init_mode=random \
        run.pretrained_checkpoint=null \
        extract.output_dir=outputs/embeddings/008_random_cwt_cnn
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.utils import get_class, instantiate
from omegaconf import DictConfig, OmegaConf
from rich.logging import RichHandler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from foundry.config_resolvers import register_resolvers
from foundry.data.datamodules.base import normalize_data_config
from foundry.seed import set_seed
from foundry.training.pretrained import TransferMode, load_pretrained_weights

logger = logging.getLogger(__name__)


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


def _build_model_and_data(cfg: DictConfig):
    """Build model and datamodule from config (mirrors main.py logic)."""
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


@torch.no_grad()
def extract_embeddings(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract backbone latent embeddings and corresponding sleep stage labels.

    Returns:
        (embeddings, labels) where embeddings is (N, embed_dim) and
        labels is (N,) with integer sleep stage classes.
    """
    model.eval()
    model.to(device)

    extractor = LatentExtractor(model)
    extractor.register()

    all_embeddings = []
    all_labels = []

    num_batches = 0
    for batch in tqdm(dataloader, desc="Extracting embeddings"):
        if max_batches is not None and num_batches >= max_batches:
            break

        target_values = batch.pop("target_values")
        batch.pop("target_weights", None)
        batch.pop("session_id", None)
        batch.pop("absolute_start", None)
        batch.pop("eval_mask", None)

        batch_device = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        model(**batch_device, unpack_output=False)

        latents = extractor.latents
        pooled = latents.mean(dim=1)
        all_embeddings.append(pooled.cpu().numpy())

        labels = None
        for task_name, targets in target_values.items():
            if "sleep" in task_name:
                labels = targets.numpy()
                break

        if labels is None:
            labels = list(target_values.values())[0].numpy()

        all_labels.append(labels)
        num_batches += 1

    extractor.remove()

    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    valid_mask = labels >= 0
    embeddings = embeddings[valid_mask]
    labels = labels[valid_mask]

    return embeddings, labels


def compute_visualizations(
    embeddings: np.ndarray,
    labels: np.ndarray,
    output_dir: Path,
    class_names: list[str] | None = None,
):
    """Compute t-SNE, PCA, and silhouette scores; save results."""
    import matplotlib.pyplot as plt

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

    OmegaConf.resolve(cfg.run)

    logger.info("Building model and data...")
    model, datamodule = _build_model_and_data(cfg)

    _initialize_vocabs(model, datamodule)
    _load_checkpoint_if_needed(model, cfg)

    val_loader = datamodule.val_dataloader()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(
        "Extracting embeddings (device=%s, max_batches=%s)...",
        device,
        max_batches,
    )
    embeddings, labels = extract_embeddings(
        model, val_loader, device, max_batches
    )
    logger.info("Extracted %d samples with dim=%d", *embeddings.shape)

    task_configs = model.task_configs
    class_names = None
    for cfg_entry in task_configs.values():
        if cfg_entry.class_mapping is not None:
            class_names = cfg_entry.get_class_names()
            break

    compute_visualizations(embeddings, labels, output_dir, class_names)


if __name__ == "__main__":
    register_resolvers()
    main()
