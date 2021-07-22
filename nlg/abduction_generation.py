import torch
from transformers import AutoTokenizer, AutoModelForPreTraining

from config import ModelConfig

params = ModelConfig.from_json("../event2mind_transformer/config.json")

SPECIAL_TOKENS = {'pad_token': '[PAD]',
                  'bos_token': '[BOS]',
                  'eos_token': '[EOS]',
                  'unk_token': '[UNK]',
                  'sep_token': '[SEP]'}

path = "/text2text_generation/checkpoints/checkpoint-3310/"
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForPreTraining.from_pretrained(path)

model.to(params.device)

#input_text = "It is PersonY's favorite color"
observation1 = "Ray drive his car on a steep mountain road."
observation2 = "Ray was fine but his car was totaled."
prompt = SPECIAL_TOKENS['bos_token'] + observation1 + SPECIAL_TOKENS['sep_token'] + observation2 + SPECIAL_TOKENS['sep_token']
generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(params.device)

model.eval()


sample_outputs = model.generate(generated,
                                do_sample=False,
                                min_length=20,
                                max_length=params.max_sequence_length,
                                # top_k=30,
                                # top_p=0.7,
                                # temperature=0.9,
                                # repetition_penalty=2.0,
                                num_return_sequences=1
                                )

text = tokenizer.decode(sample_outputs[0], skip_special_tokens=True)
print(text[len(prompt):])