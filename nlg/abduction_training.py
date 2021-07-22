import os

from transformers.modeling_outputs import MultipleChoiceModelOutput

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


from datasets import load_dataset

from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForPreTraining

from config import ModelConfig

SPECIAL_TOKENS = {'pad_token': '[PAD]',
                  'bos_token': '[BOS]',
                  'eos_token': '[EOS]',
                  'unk_token': '[UNK]',
                  'sep_token': '[SEP]'}


params = ModelConfig.from_json("abduction_config.json")

def preprocess_function(examples):
    correct_hypothesis = examples["hypothesis_"+str(examples['label'])]
    input_text = SPECIAL_TOKENS['bos_token'] + examples['observation_1'] + SPECIAL_TOKENS['sep_token'] + examples['observation_2'] + SPECIAL_TOKENS['sep_token'] + correct_hypothesis
    encoding_dict = tokenizer(input_text,
                              truncation=True,
                              max_length=params.max_sequence_length,
                              pad_to_max_length=True)

    return {
        "label": encoding_dict["input_ids"],
        "input_ids": encoding_dict["input_ids"],
        "attention_mask": encoding_dict["attention_mask"]
    }


def freeze_lower_layers(model, num_unfreeze_last_layers=0):
    for parameter in model.parameters():
        parameter.requires_grad = False

    for i, m in enumerate(model.transformer.h):
        #Only un-freeze the last n transformer blocks
        if i+1 > len(model.transformer.h) - num_unfreeze_last_layers:
            for parameter in m.parameters():
                parameter.requires_grad = True

    for parameter in model.transformer.ln_f.parameters():
        parameter.requires_grad = True

    for parameter in model.lm_head.parameters():
        parameter.requires_grad = True

    return model



dataset = load_dataset(params.dataset_name)

tokenizer = AutoTokenizer.from_pretrained(params.model_checkpoint, use_fast=True)
tokenizer.add_special_tokens(SPECIAL_TOKENS)
encoded_dataset = dataset.map(preprocess_function)#, batched=True)


model = AutoModelForPreTraining.from_pretrained(params.model_checkpoint).to(params.device)
model.resize_token_embeddings(len(tokenizer))
#model = freeze_lower_layers(model, params.num_unfreeze_last_layers)


args = TrainingArguments(
    params.checkpoint_dir,
    evaluation_strategy="epoch",
    learning_rate=params.learning_rate,
    per_device_train_batch_size=params.batch_size,
    per_device_eval_batch_size=params.batch_size,
    gradient_accumulation_steps=params.batch_update,
    num_train_epochs=params.num_train_epochs,
    weight_decay=params.weight_decay,
    load_best_model_at_end=True,
    fp16=True,
    fp16_opt_level=params.apex_opt_level,
    warmup_steps=params.warmup_steps,
)

trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    tokenizer=tokenizer
)

trainer.train()

trainer.evaluate()


