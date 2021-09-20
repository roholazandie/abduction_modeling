# import transformers
# from transformers import AutoTokenizer
# from datasets import load_dataset, load_metric, DatasetDict
# from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer
# from dataclasses import dataclass
# from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
# from typing import Optional, Union
# from config import ModelConfig
# import numpy as np
# import torch
#
# SPECIAL_TOKENS = {'pad_token': '[PAD]',
#                   'bos_token': '[BOS]',
#                   'eos_token': '[EOS]',
#                   'unk_token': '[UNK]',
#                   'sep_token': '[SEP]'}
#
#
# def compute_metrics(eval_predictions):
#     predictions, label_ids = eval_predictions
#     preds = np.argmax(predictions, axis=1)
#     return {"accuracy": (preds == label_ids).astype(np.float32).mean().item()}
#
#
# @dataclass
# class DataCollatorForMultipleChoice:
#     """
#     Data collator that will dynamically pad the inputs for multiple choice received.
#     """
#
#     tokenizer: PreTrainedTokenizerBase
#     padding: Union[bool, str, PaddingStrategy] = True
#     max_length: Optional[int] = None
#     pad_to_multiple_of: Optional[int] = None
#
#     def __call__(self, features):
#         label_name = "label" if "label" in features[0].keys() else "labels"
#         labels = [feature.pop(label_name)-1 for feature in features]
#         batch_size = len(features)
#         num_choices = len(features[0]["input_ids"])
#         flattened_features = [[{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features]
#         flattened_features = sum(flattened_features, [])
#
#         batch = self.tokenizer.pad(
#             flattened_features,
#             padding=self.padding,
#             max_length=self.max_length,
#             pad_to_multiple_of=self.pad_to_multiple_of,
#             return_tensors="pt",
#         )
#
#         # Un-flatten
#         batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
#         # Add back labels
#         batch["labels"] = torch.tensor(labels, dtype=torch.int64)
#         return batch
#
#
# def preprocess_function(examples):
#     correct_hypothesis = examples["hypothesis_" + str(examples['label'])]
#     wrong_hypothesis = examples["hypothesis_" + str(1 if examples['label']==2 else 2)]
#
#     causes_obs1 = examples['selected_causes_obs1'].strip()
#     xeffect_obs1 = examples['selected_xeffect_obs1'].strip()
#     xintent_obs1 = examples['selected_xintent_obs1'].strip()
#     xreact_obs1 = examples['selected_xreact_obs1'].strip()
#     xwant_obs1 = examples['selected_xwant_obs1'].strip()
#     isbefore_obs1 = examples['selected_isbefore_obs1'].strip()
#
#     oeffect_obs2 = examples['selected_oeffect_obs2'].strip()
#     oreact_obs2 = examples['selected_oreact_obs2'].strip()
#     owant_obs2 = examples['selected_owant_obs2'].strip()
#     xreason_obs2 = examples['selected_xreason_obs2'].strip()
#     isafter_obs2 = examples['selected_isafter_obs2'].strip()
#
#     first_sentence = [SPECIAL_TOKENS['bos_token'] + \
#                       causes_obs1 + \
#                       SPECIAL_TOKENS['sep_token'] + \
#                       xeffect_obs1 + \
#                       SPECIAL_TOKENS['sep_token'] + \
#                       xintent_obs1 + \
#                       SPECIAL_TOKENS['sep_token'] + \
#                       xreact_obs1 + \
#                       SPECIAL_TOKENS['sep_token'] + \
#                       xwant_obs1 + \
#                       SPECIAL_TOKENS['sep_token'] + \
#                       isbefore_obs1 + \
#                       SPECIAL_TOKENS['sep_token'] + \
#                       oeffect_obs2 + \
#                       SPECIAL_TOKENS['sep_token'] + \
#                       oreact_obs2 + \
#                       SPECIAL_TOKENS['sep_token'] + \
#                       owant_obs2 + \
#                       SPECIAL_TOKENS['sep_token'] + \
#                       xreason_obs2 + \
#                       SPECIAL_TOKENS['sep_token'] + \
#                       isafter_obs2 + \
#                       SPECIAL_TOKENS['sep_token'] + \
#                       examples['observation_1'] + \
#                       SPECIAL_TOKENS['sep_token'] + \
#                       examples['observation_2'] + \
#                       SPECIAL_TOKENS['sep_token']] * 2
#
#     second_sentence = [correct_hypothesis, wrong_hypothesis]
#
#     # Tokenize
#     tokenized_examples = tokenizer(first_sentence, second_sentence, truncation=True)
#     # Un-flatten
#     return tokenized_examples
#
#
# params = ModelConfig.from_json("abductive_nli_config.json")
#
# model_checkpoint = "bert-base-uncased"
#
# dataset = DatasetDict.load_from_disk(params.dataset_name)
#
# model = AutoModelForMultipleChoice.from_pretrained(params.model_checkpoint)
# tokenizer = AutoTokenizer.from_pretrained(params.model_checkpoint, use_fast=True)
#
#
# encoded_datasets = dataset.map(preprocess_function, batched=False)
#
#
# args = TrainingArguments(
#     params.checkpoint_dir,
#     evaluation_strategy = "epoch",
#     learning_rate=params.learning_rate,
#     per_device_train_batch_size=params.batch_size,
#     per_device_eval_batch_size=params.batch_size,
#     num_train_epochs=params.num_train_epochs,
#     weight_decay=params.weight_decay,
#     report_to="wandb",
#     run_name="gpt2_nli_newparams"
# )
#
# accepted_keys = ["input_ids", "attention_mask", "label"]
# features = [{k: v for k, v in encoded_datasets["train"][i].items() if k in accepted_keys} for i in range(10)]
# batch = DataCollatorForMultipleChoice(tokenizer)(features)
#
# trainer = Trainer(
#     model,
#     args,
#     train_dataset=encoded_datasets["train"],
#     eval_dataset=encoded_datasets["validation"],
#     tokenizer=tokenizer,
#     data_collator=DataCollatorForMultipleChoice(tokenizer),
#     compute_metrics=compute_metrics,
# )
#
# trainer.train()
#
# trainer.evaluate()


import transformers
from transformers import AutoTokenizer
from datasets import load_dataset, load_metric, DatasetDict
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union
from config import ModelConfig
import numpy as np
import torch

SPECIAL_TOKENS = {'pad_token': '[PAD]',
                  'bos_token': '[BOS]',
                  'eos_token': '[EOS]',
                  'unk_token': '[UNK]',
                  'sep_token': '[SEP]'}


def compute_metrics(eval_predictions):
    predictions, label_ids = eval_predictions
    preds = np.argmax(predictions, axis=1)
    return {"accuracy": (preds == label_ids).astype(np.float32).mean().item()}


@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) - 1 for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [[{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in
                              features]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch


def preprocess_function(examples):

    context = [[c1+c2+c3+c4+c5+c6+c7+c8+c9+c10+ c11+ c12+ c13+ c14]*2 for c1,c2,c3,c4,c5,c6,c7,c8,c9, c10, c11, c12, c13, c14 in zip(examples['selected_causes_obs1'],
                                                                                                                                     examples['selected_xeffect_obs1'],
                                                                                                                                     examples['selected_xintent_obs1'],
                                                                                                                                     examples['selected_xreact_obs1'],
                                                                                                                                     examples['selected_xwant_obs1'],
                                                                                                                                     examples['selected_isbefore_obs1'],
                                                                                                                                     examples['selected_oeffect_obs2'],
                                                                                                                                     examples['selected_oeffect_obs2'],
                                                                                                                                     examples['selected_oreact_obs2'],
                                                                                                                                     examples['selected_owant_obs2'],
                                                                                                                                     examples['selected_xreason_obs2'],
                                                                                                                                     examples['selected_isafter_obs2'],
                                                                                                                                     examples['observation_1'],
                                                                                                                                     examples['observation_2']
                                                                                                                                     )]


    hypotheses = [[h1, h2] for h1, h2 in zip(examples["hypothesis_1"], examples["hypothesis_2"])]

    # Flatten everything
    context = sum(context, [])
    hypotheses = sum(hypotheses, [])

    # Tokenize
    tokenized_examples = tokenizer(context, hypotheses, truncation=True)
    # Un-flatten
    return {k: [v[i:i + 2] for i in range(0, len(v), 2)] for k, v in tokenized_examples.items()}


params = ModelConfig.from_json("abductive_nli_config.json")

model_checkpoint = "bert-base-uncased"
batch_size = 5

dataset = DatasetDict.load_from_disk(params.dataset_name)

model = AutoModelForMultipleChoice.from_pretrained(params.model_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(params.model_checkpoint, use_fast=True)

encoded_datasets = dataset.map(preprocess_function, batched=True)

args = TrainingArguments(
    params.checkpoint_dir,
    evaluation_strategy="epoch",
    learning_rate=params.learning_rate,
    per_device_train_batch_size=params.batch_size,
    per_device_eval_batch_size=params.batch_size,
    num_train_epochs=params.num_train_epochs,
    weight_decay=params.weight_decay,
    report_to="wandb",
    run_name="gpt2_nli_newparams"
)


trainer = Trainer(
    model,
    args,
    train_dataset=encoded_datasets["train"],
    eval_dataset=encoded_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForMultipleChoice(tokenizer),
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.evaluate()