import numpy as np
import torch
import argparse
from datasets import load_dataset, DatasetDict
from sentence_transformers import SentenceTransformer

def get_selected_relation(relation, observation):
    observation_embedding = model.encode(observation)
    relation_embeddings = model.encode(relation)
    i = np.argmax(np.dot(relation_embeddings, observation_embedding))
    selected_relation = relation[i]
    return selected_relation

def get_random_relation(relation, observation):
    i = np.random.randint(0, len(relation))
    return relation[i]


model = SentenceTransformer('sentence-transformers/paraphrase-xlm-r-multilingual-v1')


def add_first_relations(example):
    new_example = example.copy()
    mapping_keys = {"selected_causes_obs1": "desires_obs1",
                    "selected_xeffect_obs1": "desires_obs2",
                    "selected_xintent_obs1": "hassubevent_obs1",
                    "selected_xreact_obs1": "hassubevent_obs2",
                    "selected_xwant_obs1": "isfilledby_obs1",
                    "selected_isbefore_obs1": "isfilledby_obs2",
                    "selected_oeffect_obs2": "motivatedbygoal_obs1",
                    "selected_oreact_obs2": "motivatedbygoal_obs2",
                    "selected_owant_obs2": "xneed_obs1",
                    #"selected_xreason_obs2": "xneed_obs2",
                    "selected_isafter_obs2": "xneed_obs2"
                    }
    inv_map = {v: k for k, v in mapping_keys.items()}
    for key in inv_map:
        new_example[inv_map[key]] = example[key][0]
    new_example["selected_xreason_obs2"] = example["xneed_obs2"][0]

    return new_example




def add_random_relations(example):
    causes_obs1 = example['causes_obs1']
    xeffect_obs1 = example['xeffect_obs1']
    xintent_obs1 = example['xintent_obs1']
    xreact_obs1 = example['xreact_obs1']
    xwant_obs1 = example['xwant_obs1']
    isbefore_obs1 = example['isbefore_obs1']

    oeffect_obs2 = example['oeffect_obs2']
    oreact_obs2 = example['oreact_obs2']
    owant_obs2 = example['owant_obs2']
    xreason_obs2 = example['xreason_obs2']
    isafter_obs2 = example['isafter_obs2']

    observation_1 = example['observation_1']
    observation_2 = example['observation_2']

    selected_causes_obs1 = get_random_relation(causes_obs1, observation_2)
    selected_xeffect_obs1 = get_random_relation(xeffect_obs1, observation_2)
    selected_xintent_obs1 = get_random_relation(xintent_obs1, observation_2)
    selected_xreact_obs1 = get_random_relation(xreact_obs1, observation_2)
    selected_xwant_obs1 = get_random_relation(xwant_obs1, observation_2)
    selected_isbefore_obs1 = get_random_relation(isbefore_obs1, observation_2)

    selected_oeffect_obs2 = get_random_relation(oeffect_obs2, observation_1)
    selected_oreact_obs2 = get_random_relation(oreact_obs2, observation_1)
    selected_owant_obs2 = get_random_relation(owant_obs2, observation_1)
    selected_xreason_obs2 = get_random_relation(xreason_obs2, observation_1)
    selected_isafter_obs2 = get_random_relation(isafter_obs2, observation_1)

    new_example = example.copy()
    new_example["selected_causes_obs1"] = selected_causes_obs1
    new_example["selected_xeffect_obs1"] = selected_xeffect_obs1
    new_example["selected_xintent_obs1"] = selected_xintent_obs1
    new_example["selected_xreact_obs1"] = selected_xreact_obs1
    new_example["selected_xwant_obs1"] = selected_xwant_obs1
    new_example["selected_isbefore_obs1"] = selected_isbefore_obs1
    new_example["selected_oeffect_obs2"] = selected_oeffect_obs2
    new_example["selected_oreact_obs2"] = selected_oreact_obs2
    new_example["selected_owant_obs2"] = selected_owant_obs2
    new_example["selected_xreason_obs2"] = selected_xreason_obs2
    new_example["selected_isafter_obs2"] = selected_isafter_obs2


    return new_example



def add_selected_relations(example):
    causes_obs1 = example['causes_obs1']
    xeffect_obs1 = example['xeffect_obs1']
    xintent_obs1 = example['xintent_obs1']
    xreact_obs1 = example['xreact_obs1']
    xwant_obs1 = example['xwant_obs1']
    isbefore_obs1 = example['isbefore_obs1']

    oeffect_obs2 = example['oeffect_obs2']
    oreact_obs2 = example['oreact_obs2']
    owant_obs2 = example['owant_obs2']
    xreason_obs2 = example['xreason_obs2']
    isafter_obs2 = example['isafter_obs2']

    observation_1 = example['observation_1']
    observation_2 = example['observation_2']

    # we look for the common sense relations that are extracted
    # from observation_1 but compatible with observation_2
    selected_causes_obs1 = get_selected_relation(causes_obs1, observation_2)
    selected_xeffect_obs1 = get_selected_relation(xeffect_obs1, observation_2)
    selected_xintent_obs1 = get_selected_relation(xintent_obs1, observation_2)
    selected_xreact_obs1 = get_selected_relation(xreact_obs1, observation_2)
    selected_xwant_obs1 = get_selected_relation(xwant_obs1, observation_2)
    selected_isbefore_obs1 = get_selected_relation(isbefore_obs1, observation_2)
    # we look for the common sense relations that are extracted
    # from observation_2 but compatible with observation_1
    selected_oeffect_obs2 = get_selected_relation(oeffect_obs2, observation_1)
    selected_oreact_obs2 = get_selected_relation(oreact_obs2, observation_1)
    selected_owant_obs2 = get_selected_relation(owant_obs2, observation_1)
    selected_xreason_obs2 = get_selected_relation(xreason_obs2, observation_1)
    selected_isafter_obs2 = get_selected_relation(isafter_obs2, observation_1)

    new_example = example.copy()
    new_example["selected_causes_obs1"] = selected_causes_obs1
    new_example["selected_xeffect_obs1"] = selected_xeffect_obs1
    new_example["selected_xintent_obs1"] = selected_xintent_obs1
    new_example["selected_xreact_obs1"] = selected_xreact_obs1
    new_example["selected_xwant_obs1"] = selected_xwant_obs1
    new_example["selected_isbefore_obs1"] = selected_isbefore_obs1
    new_example["selected_oeffect_obs2"] = selected_oeffect_obs2
    new_example["selected_oreact_obs2"] = selected_oreact_obs2
    new_example["selected_owant_obs2"] = selected_owant_obs2
    new_example["selected_xreason_obs2"] = selected_xreason_obs2
    new_example["selected_isafter_obs2"] = selected_isafter_obs2


    return new_example


parser = argparse.ArgumentParser()
#parser.add_argument('--input_dataset', type=str, default="/media/sdc/rohola_data/abduction_modeling_datasets/nli_random_commonsense_dataset")
parser.add_argument('--input_dataset', type=str, default="/home/rohola/codes/abduction_modeling/data/nli_random_commonsense_dataset")
#parser.add_argument('--output_dataset', type=str, default="/media/sdc/rohola_data/filtered_new_abduction_augmented_dataset")
#parser.add_argument('--output_dataset', type=str, default="/media/sdc/rohola_data/abduction_modeling_datasets/nli_filtered_random_commonsense_dataset")
parser.add_argument('--output_dataset', type=str, default="/home/rohola/codes/abduction_modeling/data/nli_filtered_random_commonsense_dataset")
# parser.add_argument('--output_dataset', type=str, default="/media/sdc/rohola_data/abduction_modeling_datasets/random_nli_dataset")
args = parser.parse_args()

dataset = DatasetDict.load_from_disk(args.input_dataset)
modified_dataset = dataset.map(add_first_relations)
modified_dataset.save_to_disk(args.output_dataset)
