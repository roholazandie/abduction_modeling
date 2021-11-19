import json
import ast
import datasets
from datasets import load_dataset, Features, Value, DatasetDict


# def convert_jsonl_to_tsv(file_path, raw_data_path):
#     split = file_path.split('/')[-1].split('.')[0]
#     delimiter = '\t'
#
#     labels = open(file_path.split('.')[0]+"-labels.lst").readlines()
#
#     with open(raw_data_path + split +".tsv", 'w') as file_writer:
#         with open(file_path) as file_reader:
#             for i, json_line in enumerate(file_reader):
#                 line = json.loads(json_line)
#                 file_writer.write(line['story_id'].rstrip('\t') + delimiter +
#                                   line['obs1'].rstrip('\t') + delimiter +
#                                   line['obs2'].rstrip('\t') + delimiter +
#                                   line['hyp1'].rstrip('\t') + delimiter +
#                                   line['hyp2'].rstrip('\t') + delimiter +
#                                   labels[i])


def add_labels_to_alphanli(file_path, raw_data_path):
    split = file_path.split('/')[-1].split('.')[0]

    labels = open(file_path.split('.')[0]+"-labels.lst").readlines()

    with open(raw_data_path + split +".jsonl", 'w') as file_writer:
        with open(file_path) as file_reader:
            for i, json_line in enumerate(file_reader):
                line = json.loads(json_line)
                line['label'] = int(labels[i].strip())
                line['observation_1'] = line.pop('obs1')
                line['observation_2'] = line.pop('obs2')
                line['hypothesis_1'] = line.pop('hyp1')
                line['hypothesis_2'] = line.pop('hyp2')

                file_writer.write(json.dumps(line)+'\n')


def change_type(x):
    x['label'] = int(x['label'])
    return x

if __name__ == "__main__":
    # raw_data_path = "/home/rohola/codes/abduction_modeling/data/art_full_raw/"
    # add_labels_to_alphanli("/home/rohola/codes/abduction_modeling/data/alphanli/train.jsonl", raw_data_path)
    # add_labels_to_alphanli("/home/rohola/codes/abduction_modeling/data/alphanli/dev.jsonl", raw_data_path)
    # add_labels_to_alphanli("/home/rohola/codes/abduction_modeling/data/alphanli/test.jsonl", raw_data_path)


    # nlg_dataset = load_dataset('json', data_files={"train": "/home/rohola/codes/abduction_modeling/data/anlg/train-w-comet-preds.jsonl",
    #                                          "test": "/home/rohola/codes/abduction_modeling/data/anlg/test-w-comet-preds.jsonl",
    #                                          "validation": "/home/rohola/codes/abduction_modeling/data/anlg/dev-w-comet-preds.jsonl"
    #                                          })
    # nlg_dataset = nlg_dataset.remove_columns('comet_preds')
    # nlg_dataset = nlg_dataset.rename_column('obs1', 'observation_1')
    # nlg_dataset = nlg_dataset.rename_column('obs2', 'observation_2')
    # nlg_dataset = nlg_dataset.rename_column('hyp1', 'hypothesis_1')
    # nlg_dataset = nlg_dataset.rename_column('hyp2', 'hypothesis_2')
    # nlg_dataset = nlg_dataset.map(change_type)
    # nlg_dataset.save_to_disk("/home/rohola/codes/abduction_modeling/data/nlg_dataset")
    #
    # x = DatasetDict.load_from_disk("/home/rohola/codes/abduction_modeling/data/nlg_dataset")


    # nli_dataset = load_dataset('json', data_files={"train": "/home/rohola/codes/abduction_modeling/data/alphanli/train.jsonl",
    #                                          "test": "/home/rohola/codes/abduction_modeling/data/alphanli/test.jsonl",
    #                                          "validation": "/home/rohola/codes/abduction_modeling/data/alphanli/dev.jsonl"
    #                                          })
    # x = 1

    x = DatasetDict.load_from_disk("/home/rohola/codes/abduction_modeling/data/art_full")

    x=1