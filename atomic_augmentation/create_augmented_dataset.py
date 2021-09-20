from datasets import load_dataset, Value, Features, load_from_disk, DatasetDict
from generation_example import Comet
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--comet_model_path', type=str, default="/media/sdc/rohola_data/abduction_checkpoints/comet_checkpoint/comet-atomic_2020_BART")
parser.add_argument('--art_dataset_path', type=str, default="/media/sdc/rohola_data/art_full")
parser.add_argument('--save_dataset_path', type=str, default="/media/sdc/rohola_data/abduction_augmented_dataset")
args = parser.parse_args()

dataset = DatasetDict.load_from_disk(args.art_dataset_path)

print("model loading ...")
#comet = Comet("/home/rohola/codes/text_nlp_hf/text2text_generation/atomic2020_transformer/official/official_checkpoint/comet-atomic_2020_BART")
comet = Comet(args.comet_model_path)
comet.model.zero_grad()
print("model loaded")


columns = ["observation_1",
           "observation_2",
           "hypothesis_1",
           "hypothesis_2",
           "label",
           "afters_obs2",
           "befores_obs1",
           "causes_obs1",
           "reason_obs2",
           "xeffects_obs1",
           "xreacts_obs1",
           "xwants_obs1",
           "xintent_obs1",
           "oeffect_obs2",
           "oreact_obs2",
           "owant_obs2"]


observation1_relations = ["isBefore", "Causes", "xEffect", "xReact", "xWant", "xIntent"]
observation2_relations = ["isAfter", "xReason", "oEffect", "oReact", "oWant"]


augmented_dataset = []

def add_relations(example):
    new_example = example.copy()
    head = example["observation_1"]
    for rel in observation1_relations:
        key = rel.lower()+"_obs1"
        new_example[key] = []
        queries = []
        query = "{} {} [GEN]".format(head, rel)
        queries.append(query)
        results = comet.generate(queries, decode_method="beam", num_generate=5)
        new_example[key].extend(results[0])


    head = example["observation_2"]
    for rel in observation2_relations:
        key = rel.lower()+"_obs2"
        new_example[key] = []
        queries = []
        query = "{} {} [GEN]".format(head, rel)
        queries.append(query)
        results = comet.generate(queries, decode_method="beam", num_generate=5)
        new_example[key].extend(results[0])

    return new_example


modified_dataset = dataset.map(add_relations)
modified_dataset.save_to_disk(args.save_dataset_path)




