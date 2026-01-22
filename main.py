from foundry.datasets import SchalkWolpawPhysionet2009

processed_dir = "./data/processed/"

physionet = SchalkWolpawPhysionet2009(
    root=processed_dir,
    task_type="LeftRightImagery",
    fold_number=0,
    fold_type="inter-subject",
)

print(physionet)
print(physionet.get_sampling_intervals(split="train"))
