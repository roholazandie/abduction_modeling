from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import json



@dataclass
class ModelConfig:
    model_checkpoint: str
    dataset_name: str
    checkpoint_dir: str
    batch_size: int
    batch_update: int
    learning_rate: float
    max_sequence_length: int
    num_train_epochs: int
    weight_decay: float
    apex_opt_level: str
    warmup_ratio: float
    num_unfreeze_last_layers: int
    device: str

    @classmethod
    def from_json(cls, json_file):
        config_json = json.load(open(json_file))
        return cls(**config_json)

    def __iter__(self):
        for attr, value in self.__dict__.items():
            yield attr, value