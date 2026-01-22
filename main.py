from brainsets.datasets.KempSleepEDF2013 import KempSleepEDF2013

dataset = KempSleepEDF2013(
    root="/network/projects/neuro-galaxy/data/processed_qc",
    config=KempSleepEDF2013.config,
    split="train",
)

print(dataset)
