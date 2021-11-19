import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


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


x = get_semantic_entailment("I am vegetarian", "I eat meat")
print(x)