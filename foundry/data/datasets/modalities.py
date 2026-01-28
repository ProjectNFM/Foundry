from torch_brain.registry import register_modality, DataType
from torch_brain.nn.loss import CrossEntropyLoss

SLEEP_STAGE_5CLASS = register_modality(
    "sleep_stage_5class",
    dim=5,
    type=DataType.MULTINOMIAL,
    timestamp_key="sleep_stages.timestamps",
    value_key="sleep_stages.values",
    loss_fn=CrossEntropyLoss(),
)

P300_TARGET = register_modality(
    "p300_target",
    dim=2,
    type=DataType.BINARY,
    timestamp_key="p300_trials.timestamps",
    value_key="p300_trials.target",
    loss_fn=CrossEntropyLoss(),
)

MOTOR_IMAGERY_5CLASS = register_modality(
    "motor_imagery_5class",
    dim=5,
    type=DataType.MULTINOMIAL,
    timestamp_key="motor_imagery_trials.timestamps",
    value_key="motor_imagery_trials.movements",
    loss_fn=CrossEntropyLoss(),
)

MOTOR_IMAGERY_LEFT_RIGHT = register_modality(
    "motor_imagery_left_right",
    dim=2,
    type=DataType.MULTINOMIAL,
    timestamp_key="motor_imagery_trials.timestamps",
    value_key="motor_imagery_trials.movements",
    loss_fn=CrossEntropyLoss(),
)

MOTOR_IMAGERY_RIGHT_FEET = register_modality(
    "motor_imagery_right_feet",
    dim=2,
    type=DataType.MULTINOMIAL,
    timestamp_key="motor_imagery_trials.timestamps",
    value_key="motor_imagery_trials.movements",
    loss_fn=CrossEntropyLoss(),
)
