from datasets import load_dataset, Value, Features, load_from_disk, DatasetDict
from generation_example import Comet
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument('--comet_model_path', type=str, default="/media/sdc/rohola_data/abduction_checkpoints/comet_checkpoint/comet-atomic_2020_BART")
parser.add_argument('--input_dataset_path', type=str, default="/media/sdc/rohola_data/abduction_modeling_datasets/nli_dataset")
parser.add_argument('--save_dataset_path', type=str, default="/media/sdc/rohola_data/abduction_modeling_datasets/nli_random_commonsense_dataset")
args = parser.parse_args()

dataset = DatasetDict.load_from_disk(args.input_dataset_path)

print("model loading ...")
comet = Comet(args.comet_model_path)
comet.model.zero_grad()
print("model loaded")

all_relations = [
    "AtLocation",
    "CapableOf",
    #"Causes",
    "CausesDesire",
    "CreatedBy",
    "DefinedAs",
    "DesireOf",
    "Desires",
    "HasA",
    "HasFirstSubevent",
    "HasLastSubevent",
    "HasPainCharacter",
    "HasPainIntensity",
    "HasPrerequisite",
    "HasProperty",
    "HasSubEvent",
    "HasSubevent",
    #"HinderedBy",
    "InheritsFrom",
    "InstanceOf",
    "IsA",
    "LocatedNear",
    "LocationOfAction",
    "MadeOf",
    "MadeUpOf",
    "MotivatedByGoal",
    "NotCapableOf",
    "NotDesires",
    "NotHasA",
    "NotHasProperty",
    "NotIsA",
    "NotMadeOf",
    "ObjectUse",
    "PartOf",
    "ReceivesAction",
    "RelatedTo",
    "SymbolOf",
    "UsedFor",
    #"isAfter",
    #"isBefore",
    "isFilledBy",
    #"oEffect",
    #"oReact",
    #"oWant",
    #"xAttr",
    #"xEffect",
    #"xIntent",
    "xNeed",
    #"xReact",
    #"xReason",
    #"xWant",
]

augmented_dataset = []

random_relations = random.sample(all_relations, 5)

def add_relations(example):
    new_example = example.copy()
    head = example["observation_1"]
    for rel in random_relations:
        key = rel.lower()+"_obs1"
        new_example[key] = []
        queries = []
        query = "{} {} [GEN]".format(head, rel)
        queries.append(query)
        results = comet.generate(queries, decode_method="beam", num_generate=5)
        new_example[key].extend(results[0])


    head = example["observation_2"]
    #random_relations = random.sample(all_relations, 5)
    for rel in random_relations:
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




