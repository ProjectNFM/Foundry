from foundry.training.task_modules import (
    ClassificationModule,
    MaskedReconstructionModule,
    RegressionModule,
)

PretrainModule = MaskedReconstructionModule

__all__ = [
    "ClassificationModule",
    "RegressionModule",
    "MaskedReconstructionModule",
    "PretrainModule",
]
