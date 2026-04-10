"""Lightning callbacks for Foundry model training."""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import lightning as L
import numpy as np
import torch
from lightning import Trainer

from foundry.core import VocabManager

_LOG = logging.getLogger(__name__)


class VocabInitializerCallback(L.Callback):
    """Callback to initialize model vocabularies from the datamodule.

    This callback handles the initialization of lazy vocabularies (e.g., session and
    channel embeddings) before training begins. It decouples vocab setup from the
    datamodule, allowing models to be reused with different datasets.

    Usage:
        trainer = Trainer(callbacks=[VocabInitializerCallback()])
    """

    def on_fit_start(
        self, trainer: Trainer, pl_module: L.LightningModule
    ) -> None:
        """Initialize vocabularies at the start of training.

        Args:
            trainer: Lightning Trainer instance.
            pl_module: Lightning module being trained.
        """
        # Check if model implements vocab initialization
        model = pl_module.model if hasattr(pl_module, "model") else pl_module

        if not isinstance(model, VocabManager):
            return

        if not model.has_lazy_vocabs():
            return

        # Get datamodule
        datamodule = trainer.datamodule
        if datamodule is None:
            raise RuntimeError(
                "VocabInitializerCallback requires a datamodule. "
                "Call trainer.fit(module, datamodule=dm) or set trainer.datamodule."
            )

        # Initialize vocabularies from datamodule or its underlying dataset
        vocab_info = {}
        dataset = getattr(datamodule, "dataset", None)

        for method_name, key in [
            ("get_recording_ids", "session_ids"),
            ("get_channel_ids", "channel_ids"),
        ]:
            if hasattr(datamodule, method_name):
                vocab_info[key] = getattr(datamodule, method_name)()
            elif dataset is not None and hasattr(dataset, method_name):
                vocab_info[key] = getattr(dataset, method_name)()

        model.initialize_vocabs(vocab_info)


_CSV_FIELDS = [
    "run_name",
    "recording_id",
    "lowcut",
    "highcut",
    "stft",
    "decimate_factor",
    "accuracy",
    "balanced_accuracy",
    "f1_macro",
    "cohen_kappa",
    "perm_pvalue",
    "perm_zscore",
    "n_classes",
    "n_valid_samples",
    "elapsed_seconds",
]


def _permutation_test(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    observed_bal_acc: float,
    n_permutations: int,
    seed: int,
) -> tuple[float, float]:
    """Permutation test on balanced accuracy (matches auditorydecoding implementation).

    Returns (p_value, z_score) where z_score = (observed - null_mean) / null_std.
    """
    from sklearn.metrics import balanced_accuracy_score

    rng = np.random.default_rng(seed)
    null_accs = np.array(
        [
            float(balanced_accuracy_score(rng.permutation(y_true), y_pred))
            for _ in range(n_permutations)
        ]
    )
    p_value = float((null_accs >= observed_bal_acc).mean())
    null_std = float(null_accs.std())
    if null_std < 1e-12:
        z_score = 0.0
    else:
        z_score = float((observed_bal_acc - null_accs.mean()) / null_std)
    return p_value, z_score


class ResultsExporterCallback(L.Callback):
    """Export evaluation metrics to a notebook-compatible CSV after training.

    Reloads the best checkpoint, runs a full validation pass, computes balanced
    accuracy and a permutation-test p-value/z-score, then writes a single-row
    CSV with the same schema as the auditorydecoding frequency_decoding_sweep.
    """

    def __init__(
        self,
        n_permutations: int = 10_000,
        permutation_seed: int = 0,
    ):
        super().__init__()
        self.n_permutations = n_permutations
        self.permutation_seed = permutation_seed

    def on_fit_end(
        self, trainer: Trainer, pl_module: L.LightningModule
    ) -> None:
        from sklearn.metrics import (
            accuracy_score,
            balanced_accuracy_score,
            cohen_kappa_score,
            f1_score,
        )

        ckpt_cb = trainer.checkpoint_callback
        if ckpt_cb is None or not getattr(ckpt_cb, "best_model_path", None):
            _LOG.warning(
                "ResultsExporterCallback: no best checkpoint found, skipping."
            )
            return

        _LOG.info("Loading best checkpoint: %s", ckpt_cb.best_model_path)
        state = torch.load(
            ckpt_cb.best_model_path, map_location=pl_module.device
        )
        pl_module.load_state_dict(state["state_dict"])
        pl_module.eval()

        all_y_true: dict[str, list[np.ndarray]] = {}
        all_y_pred: dict[str, list[np.ndarray]] = {}

        val_dl = trainer.datamodule.val_dataloader()
        with torch.no_grad():
            for batch in val_dl:
                batch = trainer.strategy.batch_to_device(batch)
                target_values = batch.pop("target_values")
                _ = batch.pop("target_weights")
                batch.pop("session_id", None)
                batch.pop("absolute_start", None)
                batch.pop("eval_mask", None)

                outputs = pl_module.model(**batch, unpack_output=False)

                for task_name, logits in outputs.items():
                    if task_name not in target_values:
                        continue
                    target = target_values[task_name]
                    if target.numel() == 0:
                        continue

                    preds = logits.argmax(dim=-1).cpu().numpy()
                    mapped = (
                        pl_module._apply_label_mapping(target, task_name)
                        .cpu()
                        .numpy()
                    )

                    all_y_true.setdefault(task_name, []).append(mapped)
                    all_y_pred.setdefault(task_name, []).append(preds)

        output_dir = Path(trainer.default_root_dir)
        recording_ids = trainer.datamodule.get_recording_ids()
        recording_id = recording_ids[0] if recording_ids else "unknown"

        for task_name in all_y_true:
            y_true = np.concatenate(all_y_true[task_name])
            y_pred = np.concatenate(all_y_pred[task_name])

            bal_acc = float(balanced_accuracy_score(y_true, y_pred))
            acc = float(accuracy_score(y_true, y_pred))
            f1 = float(
                f1_score(y_true, y_pred, average="macro", zero_division=0)
            )
            kappa = float(cohen_kappa_score(y_true, y_pred))
            n_classes = int(len(np.unique(y_true)))

            perm_p, perm_z = _permutation_test(
                y_true,
                y_pred,
                bal_acc,
                self.n_permutations,
                self.permutation_seed,
            )

            csv_path = output_dir / "results.csv"
            _LOG.info(
                "Writing results to %s  (bal_acc=%.4f, perm_p=%.4g, perm_z=%.2f)",
                csv_path,
                bal_acc,
                perm_p,
                perm_z,
            )
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            with csv_path.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
                w.writeheader()
                w.writerow(
                    {
                        "run_name": f"eegnet_{task_name}",
                        "recording_id": recording_id,
                        "lowcut": "",
                        "highcut": "",
                        "stft": "False",
                        "decimate_factor": 1,
                        "accuracy": f"{acc:.6f}",
                        "balanced_accuracy": f"{bal_acc:.6f}",
                        "f1_macro": f"{f1:.6f}",
                        "cohen_kappa": f"{kappa:.6f}",
                        "perm_pvalue": f"{perm_p:.6f}",
                        "perm_zscore": f"{perm_z:.6f}",
                        "n_classes": n_classes,
                        "n_valid_samples": len(y_true),
                        "elapsed_seconds": "0.00",
                    }
                )
