from foundry.datasets import KorczowskiBrainInvaders2014a
from torch_brain.data.sampler import RandomFixedWindowSampler


processed_dir = "./data/processed/"

korczowski = KorczowskiBrainInvaders2014a(
    root=processed_dir,
    fold_number=0,
    fold_type="inter-subject",
)

sampler = RandomFixedWindowSampler(
    sampling_intervals=korczowski.get_sampling_intervals(split="train"),
    window_length=1.0,
)

print(sampler)
