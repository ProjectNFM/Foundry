    
from auditorydecoding import NeurosoftDataset
from temporaldata import Data
from typing import Optional, Callable, Literal, Type
import numpy as np

class BalanceData:
     
    def __init__(self,
            dataset: Type[NeurosoftDataset],
            parent_split: Optional[
                Literal[
                    "train",
                    "valid",
                    "test",
                ]
            ] = "train", 
            balance_type: Optional[
                Literal[
                    "threshold",
                    "d-threshold",
                    "percentile",
                    "downsample",
                ]
            ] = None, 
            min_trials: int = 0, 
            retain_percentile: int = 0, 
            balance_seed: int = 42
            ):
        
        self.valid_splits       = ["train", "valid", "test"]
        
        self.balance_type       = balance_type
        self.min_trials         = min_trials
        self.retain_percentile  = retain_percentile
        self.balance_seed       = balance_seed

        self.valid_classes = None

        intervals = dataset.get_sampling_intervals(split=parent_split)

        all_labels = np.concatenate([
            np.asarray(iv.behavior_labels) for iv in intervals.values() if len(iv) > 0 and hasattr(iv, "behavior_labels")
        ])

        if len(all_labels) == 0:
            return intervals, set()
        
        unique_classes, counts = np.unique(all_labels, return_counts=True)

        if self.balance_type == "percentile":
            self.min_trials = np.percentile(counts, self.retain_percentile)
            self.valid_classes = set(
                unique_classes[(counts >= self.min_trials)]
            )
        elif self.balance_type in ("threshold", "d-threshold"):
            self.valid_classes = set(
                unique_classes[counts >= self.min_trials]
            )

    def modify_dataset(self, dataset):
        for split in self.valid_splits:
            intervals = dataset.get_sampling_intervals(split=split)

            # Check if we need to filter the valid intervals
            if self.valid_classes is not None:
                intervals = {
                    rid: (
                        iv.select_by_mask(
                            np.isin(np.asarray(iv.behavior_labels), list(self.valid_classes))
                        )
                        if len(iv) > 0 and hasattr(iv, "behavior_labels")
                        else iv
                    )
                    for rid, iv in intervals.items()
                }

            all_labels = np.concatenate([
                np.asarray(iv.behavior_labels) for iv in intervals.values() if len(iv) > 0 and hasattr(iv, "behavior_labels")
            ])

            if len(all_labels) > 0:
                unique_classes, counts = np.unique(all_labels, return_counts=True)

                # Enter stochastic downsample loop
                if self.balance_type in ("downsample", "d-threshold", "percentile"):
                    global_limit = int(counts.min())
                    rng = np.random.default_rng(self.balance_seed)
                    all_indices = {cls: [] for cls in unique_classes}
                    for rid, iv in intervals.items():
                        if len(iv) > 0 and hasattr(iv, "behavior_labels"):
                            for i, lbl in enumerate(iv.behavior_labels):
                                if lbl in all_indices:
                                    all_indices[lbl].append((rid, i))
                    kept = {rid: np.zeros(len(iv), dtype=bool) for rid, iv in intervals.items()}
                    for cls, idx_list in all_indices.items():
                        if len(idx_list) == 0:
                            continue
                        n = min(global_limit, len(idx_list))
                        chosen = rng.choice(len(idx_list), size=n, replace=False)
                        for i in chosen:
                            rid, local_i = idx_list[i]
                            kept[rid][local_i] = True
                    intervals = {
                        rid: iv.select_by_mask(kept[rid]) for rid, iv in intervals.items()
                    }

            dataset.set_sampling_intervals(intervals=intervals, split=split)

        return dataset