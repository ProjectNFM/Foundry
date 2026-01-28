from foundry.models import EEGModel, LinearEmbedding
from foundry.data.datamodules import PhysionetDataModule
from foundry.data.datasets import SchalkWolpawPhysionet2009
from foundry.transforms.patching import Patching


processed_dir = "./data/processed/"

eeg_model = EEGModel(
    input_embedding=LinearEmbedding(embed_dim=128),
    readout_specs=SchalkWolpawPhysionet2009.get_modality_specs(),
    embed_dim=128,
    sequence_length=1.0,
)

data_module = PhysionetDataModule(
    root=processed_dir,
    model=eeg_model,
    window_length=1.0,
    transform=Patching(patch_duration=0.1),
)
data_module.setup("fit")

sample1 = next(iter(data_module.train_dataloader()))

eeg_model(sample1)
