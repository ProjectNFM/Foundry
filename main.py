from foundry.models import EEGModel, LinearEmbedding
from foundry.data.datamodules import PhysionetDataModule
from foundry.transforms.patching import Patching

processed_dir = "./data/processed/"

eeg_model = EEGModel(
    input_embedding=LinearEmbedding(embed_dim=128),
    readout_specs=["motor_imagery_5class"],
    embed_dim=128,
    sequence_length=1.0,
    context_mode="add",
)

data_module = PhysionetDataModule(
    root=processed_dir,
    model=eeg_model,
    window_length=1.0,
    transform=Patching(patch_duration=0.1),
)
data_module.setup("fit")

sample1 = next(iter(data_module.train_dataloader()))

target_values = sample1.pop("target_values")
target_weights = sample1.pop("target_weights")
session_id = sample1.pop("session_id")
absolute_start = sample1.pop("absolute_start")
eval_mask = sample1.pop("eval_mask")

output = eeg_model(**sample1)
print(f"Output keys: {output.keys()}")
print(f"Motor imagery output shape: {output['motor_imagery_5class'].shape}")
