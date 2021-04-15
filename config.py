from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import json, yaml
import torch
import argparse

from transformers import LogitsProcessorList, StoppingCriteriaList


@dataclass
class ModelConfig:

    pretrained_model_forward: str
    pretrained_model_backward: str
    length: int
    max_length: int
    mix_rate: float
    temperature_forward: float
    temperature_backward: float
    top_k_forward: int
    top_k_backward: int
    top_p_forward: float
    top_p_backward: float
    repetition_penalty_forward: float
    repetition_penalty_backward: float
    device: str

    @classmethod
    def from_json(cls, json_file):
        config_json = json.load(open(json_file))
        return cls(**config_json)

    @classmethod
    def from_yaml(cls, yaml_file):
        config_yaml = yaml.load(open(yaml_file), yaml.FullLoader)
        return cls(**config_yaml)

    def __iter__(self):
        for attr, value in self.__dict__.items():
            yield attr, value

@dataclass
class DataConfig:
    abductive_dataset: str

    @classmethod
    def from_json(cls, json_file):
        config_json = json.load(open(json_file))
        return cls(**config_json)

    @classmethod
    def from_yaml(cls, yaml_file):
        config_yaml = yaml.load(open(yaml_file), yaml.FullLoader)
        return cls(**config_yaml)

    def __iter__(self):
        for attr, value in self.__dict__.items():
            yield attr, value



@dataclass
class GenerationSpecConfig:
    input_ids: Optional[torch.LongTensor] = None
    logits_processor: Optional[LogitsProcessorList] = None
    stopping_criteria: Optional[StoppingCriteriaList] = None
    logits_warper: Optional[LogitsProcessorList] = None
    max_length: Optional[int] = None
    model_kwargs: Dict[str, Any] = None
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    bos_token_id: Optional[int] = None