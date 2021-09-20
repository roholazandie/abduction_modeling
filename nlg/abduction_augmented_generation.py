import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForPreTraining, GPTNeoForCausalLM, GPT2Tokenizer
from datasets import load_dataset, DatasetDict
from config import ModelConfig
from tqdm import tqdm
from datasets import load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

nli_model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/nli-deberta-base')
nli_tokenizer = AutoTokenizer.from_pretrained('cross-encoder/nli-deberta-base')

def get_semantic_entailment(sent1, sent2):
    features = nli_tokenizer(sent1, sent2,
                         padding=True,
                         truncation=True,
                         return_tensors="pt")

    nli_model.eval()
    with torch.no_grad():
        scores = nli_model(**features).logits
        label_mapping = ['contradiction', 'entailment', 'neutral']
        labels = [label_mapping[score_max] for score_max in scores.argmax(dim=1)]

    return labels[0]


bleu = load_metric("bleu")
params = ModelConfig.from_json("abduction_augmented_config.json")

SPECIAL_TOKENS = {'pad_token': '[PAD]',
                  'bos_token': '[BOS]',
                  'eos_token': '[EOS]',
                  'unk_token': '[UNK]',
                  'sep_token': '[SEP]'}

tokenizer = AutoTokenizer.from_pretrained(params.checkpoint_dir)
model = AutoModelForPreTraining.from_pretrained(params.checkpoint_dir)

# tokenizer = GPT2Tokenizer.from_pretrained(params.checkpoint_dir)
# model = GPTNeoForCausalLM.from_pretrained(params.checkpoint_dir)


model.to(params.device)
model.eval()

dataset = DatasetDict.load_from_disk(params.dataset_name)

all_bleus = []

fw = open("results_test.csv", 'w')
fw.write("observation1, observation1, generated hypothesis, real hypothesis\n")

for example in dataset["validation"]:
    correct_hypothesis = example["hypothesis_"+str(example['label'])]

    causes_obs1 = example['selected_causes_obs1'].strip()
    xeffect_obs1 = example['selected_xeffect_obs1'].strip()
    xintent_obs1 = example['selected_xintent_obs1'].strip()
    xreact_obs1 = example['selected_xreact_obs1'].strip()
    xwant_obs1 = example['selected_xwant_obs1'].strip()
    
    oeffect_obs2 = example['selected_oeffect_obs2'].strip()
    oreact_obs2 = example['selected_oreact_obs2'].strip()
    owant_obs2 = example['selected_owant_obs2'].strip()
    xreason_obs2 = example['selected_xreason_obs2'].strip()

    observation_1 = example['observation_1']
    observation_2 = example['observation_2']

    prompt_text = SPECIAL_TOKENS['bos_token'] + \
                          causes_obs1+ \
                          SPECIAL_TOKENS['sep_token'] + \
                          xeffect_obs1 + \
                          SPECIAL_TOKENS['sep_token'] + \
                          xintent_obs1 + \
                          SPECIAL_TOKENS['sep_token'] + \
                          xreact_obs1 + \
                          SPECIAL_TOKENS['sep_token'] + \
                          xwant_obs1 + \
                          SPECIAL_TOKENS['sep_token'] + \
                          oeffect_obs2 + \
                          SPECIAL_TOKENS['sep_token'] + \
                          oreact_obs2 + \
                          SPECIAL_TOKENS['sep_token'] + \
                          owant_obs2 + \
                          SPECIAL_TOKENS['sep_token'] + \
                          xreason_obs2 + \
                          SPECIAL_TOKENS['sep_token'] + \
                          observation_1 + \
                          SPECIAL_TOKENS['sep_token'] + \
                          observation_2 + \
                          SPECIAL_TOKENS['sep_token']
    
    
    prompt = torch.tensor(tokenizer.encode(prompt_text)).unsqueeze(0).to(params.device)
    sample_outputs = model.generate(prompt,
                                    do_sample=False,
                                    min_length=10,
                                    max_length=params.max_sequence_length,
                                    # top_k=30,
                                    # top_p=0.7,
                                    # temperature=0.9,
                                    # repetition_penalty=2.0,
                                    num_return_sequences=1
                                    )
    
    text = tokenizer.decode(sample_outputs[0], skip_special_tokens=False)
    predicted_text = text[len(prompt_text):].replace('[PAD]', '')
    predicted_sentences = list(set(predicted_text.split('.')[:-1]))

    #evaluate which sentences entail from obs1 or at least neutral,
    # remove the ones that have contradiction with either of obs1 and obs2
    filtered_predicted_sentences = []
    for predicted_sentence in predicted_sentences:
        obs1_to_pred = False
        pred_to_obs2 = False
        label = get_semantic_entailment(observation_1, predicted_sentence)
        if label == "contradiction":
            obs1_to_pred = True

        label = get_semantic_entailment(predicted_sentence, observation_2)
        if label == "contradiction":
            pred_to_obs2 = True

        if not (obs1_to_pred and pred_to_obs2):
            filtered_predicted_sentences.append(predicted_sentence)


    if len(filtered_predicted_sentences) == 0:
        filtered_predicted_sentences = predicted_sentences[0]


    filtered_predicted_text =  ". ".join(filtered_predicted_sentences)

    prediction = [filtered_predicted_text.split()]
    reference = [[correct_hypothesis.split()]]


    results = bleu.compute(predictions=prediction, references=reference)
    #print(results)
    all_bleus.append(results['bleu'])

    print(f"observation1: {observation_1}")
    print(f"observation2: {observation_2}")
    print(f"generated hypothesis: {text[len(prompt_text):].replace('[PAD]', '').split('.')[:3]}")
    print(f"real hypothesis: {correct_hypothesis}")
    print("###################################")

    generated_hypothesis = '. '.join(text[len(prompt_text):].replace('[PAD]', '').replace(',', '').split('.')[:3])

    fw.write(observation_1.replace(',', '') + "," + observation_2.replace(',', '') + "," + generated_hypothesis + "," + correct_hypothesis.replace(',', '')+'\n')

fw.close()
print(np.mean(all_bleus))
print(np.max(all_bleus))
