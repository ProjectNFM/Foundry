from foundry.datasets import KorczowskiBrainInvaders2014a
from foundry.models import EEGModel, LinearEmbedding

processed_dir = "./data/processed/"

korczowski = KorczowskiBrainInvaders2014a(
    root=processed_dir,
    fold_number=0,
    fold_type="inter-subject",
)

eeg_model = EEGModel(
    input_embedding=LinearEmbedding(embed_dim=128),
    readout_specs=KorczowskiBrainInvaders2014a.get_modality_specs(),
    embed_dim=128,
    sequence_length=1.0,
)
