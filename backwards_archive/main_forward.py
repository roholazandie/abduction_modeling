from transformers import AutoTokenizer, AutoModel, GPT2Tokenizer, GPT2LMHeadModel
import torch
import numpy as np


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2").to('cuda:0')


# inputs = tokenizer("Hello world!", return_tensors="pt", add_special_tokens=True)
# outputs = model(**inputs)
# print(outputs)

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)


prompt_text = "Ray drive his car on a steep mountain road."
input_ids = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt").to('cuda:0')


length = 30
repetition_penalty = 1.0
num_return_sequences = 5
stop_token = None

output_sequences = model.generate(
    input_ids=input_ids,
    max_length=length + len(input_ids[0]),
    temperature=1.0,
    top_k=0,
    top_p=0.9,
    repetition_penalty=repetition_penalty,
    do_sample=True,
    num_return_sequences=num_return_sequences,
)



generated_sequences = []
for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
    print("=== GENERATED SEQUENCE {} ===".format(generated_sequence_idx + 1))
    generated_sequence = generated_sequence.tolist()

    # Decode text
    text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

    # Remove all text after the stop token
    text = text[: text.find(stop_token) if stop_token else None]


    total_sequence = (prompt_text + text[len(tokenizer.decode(input_ids[0], clean_up_tokenization_spaces=True)) :])

    generated_sequences.append(total_sequence)
    print(total_sequence)