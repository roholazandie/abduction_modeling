import os
from datasets import load_dataset, DatasetDict
from transformers.modeling_outputs import MultipleChoiceModelOutput
from transformers import GPTNeoForCausalLM, GPT2Tokenizer

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from datasets import load_dataset

from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForPreTraining

from config import ModelConfig

SPECIAL_TOKENS = {'pad_token': '[PAD]',
                  'bos_token': '[BOS]',
                  'eos_token': '[EOS]',
                  'unk_token': '[UNK]',
                  'sep_token': '[SEP]'}

params = ModelConfig.from_json("abduction_augmented_config.json")


# def compute_metrics(eval_preds):
#     print(eval_preds)


def preprocess_function(examples):
    correct_hypothesis = examples["hypothesis_" + str(examples['label'])]
    causes_obs1 = examples['selected_causes_obs1'].strip()
    xeffect_obs1 = examples['selected_xeffect_obs1'].strip()
    xintent_obs1 = examples['selected_xintent_obs1'].strip()
    xreact_obs1 = examples['selected_xreact_obs1'].strip()
    xwant_obs1 = examples['selected_xwant_obs1'].strip()
    isbefore_obs1 = examples['selected_isbefore_obs1'].strip()

    oeffect_obs2 = examples['selected_oeffect_obs2'].strip()
    oreact_obs2 = examples['selected_oreact_obs2'].strip()
    owant_obs2 = examples['selected_owant_obs2'].strip()
    xreason_obs2 = examples['selected_xreason_obs2'].strip()
    isafter_obs2 = examples['selected_isafter_obs2'].strip()

    input_text = SPECIAL_TOKENS['bos_token'] + \
                 causes_obs1 + \
                 SPECIAL_TOKENS['sep_token'] + \
                 xeffect_obs1 + \
                 SPECIAL_TOKENS['sep_token'] + \
                 xintent_obs1 + \
                 SPECIAL_TOKENS['sep_token'] + \
                 xreact_obs1 + \
                 SPECIAL_TOKENS['sep_token'] + \
                 xwant_obs1 + \
                 SPECIAL_TOKENS['sep_token'] + \
                 isbefore_obs1 + \
                 SPECIAL_TOKENS['sep_token'] + \
                 oeffect_obs2 + \
                 SPECIAL_TOKENS['sep_token'] + \
                 oreact_obs2 + \
                 SPECIAL_TOKENS['sep_token'] + \
                 owant_obs2 + \
                 SPECIAL_TOKENS['sep_token'] + \
                 xreason_obs2 + \
                 SPECIAL_TOKENS['sep_token'] + \
                 isafter_obs2 + \
                 SPECIAL_TOKENS['sep_token'] + \
                 examples['observation_1'] + \
                 SPECIAL_TOKENS['sep_token'] + \
                 examples['observation_2'] + \
                 SPECIAL_TOKENS['sep_token'] + \
                 correct_hypothesis

    encoding_dict = tokenizer(input_text,
                              truncation=True,
                              max_length=params.max_sequence_length,
                              pad_to_max_length=True)

    return {
        "label": encoding_dict["input_ids"],
        "input_ids": encoding_dict["input_ids"],
        "attention_mask": encoding_dict["attention_mask"]
    }


dataset = DatasetDict.load_from_disk(params.dataset_name)

#tokenizer = AutoTokenizer.from_pretrained(params.model_checkpoint, use_fast=True)
#model = AutoModelForPreTraining.from_pretrained(params.model_checkpoint).to(params.device)

model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")#.to(params.device)
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")

tokenizer.add_special_tokens(SPECIAL_TOKENS)
encoded_dataset = dataset.map(preprocess_function)  # , batched=True)


model.resize_token_embeddings(len(tokenizer))


args = TrainingArguments(
    params.checkpoint_dir,
    per_device_train_batch_size=params.batch_size,
    per_device_eval_batch_size=params.batch_size,
    num_train_epochs=params.num_train_epochs,
    weight_decay=0.01,
    warmup_steps=100,
    logging_dir="/media/sdb4Tb/rohola_data",
    logging_steps=5000,
    save_steps=5000,
)



# args = TrainingArguments(
#     params.checkpoint_dir,
#     evaluation_strategy="epoch",
#     save_strategy="epoch",
#     learning_rate=params.learning_rate,
#     per_device_train_batch_size=params.batch_size,
#     per_device_eval_batch_size=params.batch_size,
#     gradient_accumulation_steps=params.batch_update,
#     num_train_epochs=params.num_train_epochs,
#     weight_decay=params.weight_decay,
#     load_best_model_at_end=True,
#     fp16=True,
#     fp16_opt_level=params.apex_opt_level,
#     warmup_steps=params.warmup_steps,
# )

trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    tokenizer=tokenizer,
    #compute_metrics=compute_metrics
)

trainer.train()

trainer.evaluate()
