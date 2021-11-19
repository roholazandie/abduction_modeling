import datasets
import pickle
import numpy as np

# bertscore = datasets.load_metric("bertscore")
# bleurt = datasets.load_metric("bleurt")
# bleu = datasets.load_metric("bleu")
# ter = datasets.load_metric("ter")
# meteor = datasets.load_metric("meteor")
# rouge = datasets.load_metric("rouge")
#
# bert_score_results = []
# references = []
# predictions = []
# with open("../data/results/evalset_all_results.csv") as file_reader:
#     for i, line in enumerate(file_reader):
#         if i == 0:
#             continue
#
#         items = line.split(',')
#         real_hypothesis = items[3]
#
#         comet_model = items[4].strip("'").strip("\"").strip()
#
#         try:
#             abduction_gpt_lg_model = items[5].strip("'").strip("\"").strip()
#         except:
#             pass
#         abduction_gpt_md_model = items[6].strip("'").strip("\"").strip()
#         abduction_gpt_sm_model = items[7].strip("'").strip("\"").split('.')[0].strip()
#
#         if real_hypothesis.strip() == "" or comet_model.strip() == "" or abduction_gpt_sm_model.strip() == "" or abduction_gpt_md_model.strip() == "" or abduction_gpt_sm_model.strip() == "":
#             continue
#
#         references.append([real_hypothesis, real_hypothesis, real_hypothesis, real_hypothesis])
#         predictions.append([comet_model, abduction_gpt_lg_model, abduction_gpt_md_model, abduction_gpt_sm_model])
#
# comet_blue = bleu.compute(predictions=[x[0].split() for x in predictions],
#                           references=[[x[0].split()] for x in references])
# lg_bleu = bleu.compute(predictions=[x[1].split() for x in predictions],
#                        references=[[x[1].split()] for x in references])
# md_bleu = bleu.compute(predictions=[x[2].split() for x in predictions],
#                        references=[[x[2].split()] for x in references])
# sd_bleu = bleu.compute(predictions=[x[3].split() for x in predictions],
#                        references=[[x[3].split()] for x in references])
#
#
# ##########bert
# bert_score_dict = bertscore.compute(predictions=[x[0] for x in predictions],
#                                     references=[x[0] for x in references],
#                                     lang="en")
# comet_bert = {"precision": np.mean(bert_score_dict['precision']),
#               "recall": np.mean(bert_score_dict['recall']),
#               "f1": np.mean(bert_score_dict['f1'])}
#
# bert_score_dict = bertscore.compute(predictions=[x[1] for x in predictions],
#                                     references=[x[1] for x in references],
#                                     lang="en")
# lg_bert = {"precision": np.mean(bert_score_dict['precision']),
#            "recall": np.mean(bert_score_dict['recall']),
#            "f1": np.mean(bert_score_dict['f1'])}
#
# bert_score_dict = bertscore.compute(predictions=[x[2] for x in predictions],
#                                     references=[x[2] for x in references],
#                                     lang="en")
# md_bert = {"precision": np.mean(bert_score_dict['precision']),
#            "recall": np.mean(bert_score_dict['recall']),
#            "f1": np.mean(bert_score_dict['f1'])}
#
# bert_score_dict = bertscore.compute(predictions=[x[3] for x in predictions],
#                                     references=[x[3] for x in references],
#                                     lang="en")
# sd_bert = {"precision": np.mean(bert_score_dict['precision']),
#            "recall": np.mean(bert_score_dict['recall']),
#            "f1": np.mean(bert_score_dict['f1'])}
#
#
# ################bluert
# bluert_score_dict = bleurt.compute(predictions=[x[0] for x in predictions], references=[x[0] for x in references])
# comet_bleurt = np.mean(bluert_score_dict['scores'])
#
# bluert_score_dict = bleurt.compute(predictions=[x[1] for x in predictions], references=[x[1] for x in references])
# lg_bleurt = np.mean(bluert_score_dict['scores'])
#
# bluert_score_dict = bleurt.compute(predictions=[x[2] for x in predictions], references=[x[2] for x in references])
# md_bleurt = np.mean(bluert_score_dict['scores'])
#
# bluert_score_dict = bleurt.compute(predictions=[x[3] for x in predictions], references=[x[3] for x in references])
# sd_bleurt = np.mean(bluert_score_dict['scores'])
#
# #############ter
# comet_ter = ter.compute(predictions=[x[0].split() for x in predictions],
#                           references=[[x[0].split()] for x in references])
# lg_ter = ter.compute(predictions=[x[1].split() for x in predictions],
#                        references=[[x[1].split()] for x in references])
# md_ter = ter.compute(predictions=[x[2].split() for x in predictions],
#                        references=[[x[2].split()] for x in references])
# sd_ter = ter.compute(predictions=[x[3].split() for x in predictions],
#                        references=[[x[3].split()] for x in references])
#
# ##########meteor
# meteor_score_dict = meteor.compute(predictions=[x[0] for x in predictions], references=[x[0] for x in references])
# comet_meteor = meteor_score_dict['meteor']
#
# meteor_score_dict = meteor.compute(predictions=[x[1] for x in predictions], references=[x[1] for x in references])
# lg_meteor = meteor_score_dict['meteor']
#
# meteor_score_dict = meteor.compute(predictions=[x[2] for x in predictions], references=[x[2] for x in references])
# md_meteor = meteor_score_dict['meteor']
#
# meteor_score_dict = meteor.compute(predictions=[x[3] for x in predictions], references=[x[3] for x in references])
# sd_meteor = meteor_score_dict['meteor']
#
# ########rouge
# rouge_score_dict = rouge.compute(predictions=[x[0] for x in predictions], references=[x[0] for x in references])
# comet_rouge =rouge_score_dict
#
# rouge_score_dict = rouge.compute(predictions=[x[1] for x in predictions], references=[x[1] for x in references])
# lg_rouge = rouge_score_dict
#
# rouge_score_dict = rouge.compute(predictions=[x[2] for x in predictions], references=[x[2] for x in references])
# md_rouge = rouge_score_dict
#
# rouge_score_dict = rouge.compute(predictions=[x[3] for x in predictions], references=[x[3] for x in references])
# sd_rouge = rouge_score_dict
#
#
# results = {"bertscore": {"comet": comet_bert, "lg": lg_bert, "md": md_bert, "sd": sd_bert},
#            "bleurt": {"comet": comet_bleurt, "lg": lg_bleurt, "md": md_bleurt, "sd": sd_bleurt},
#            "bleu": {"comet": comet_blue, "lg": lg_bleu, "md": md_bleu, "sd": sd_bleu},
#            "ter": {"comet": comet_ter, "lg": lg_ter, "md": md_ter, "sd": sd_ter},
#            "meteor": {"comet": comet_meteor, "lg": lg_meteor, "md": md_meteor, "sd": sd_meteor},
#            "rouge": {"comet": comet_rouge, "lg": lg_rouge, "md": md_rouge, "sd": sd_rouge},
#            }
#
#
# # print(bert_score_results)
# with open('results_eval.pkl', 'wb') as f:
#     pickle.dump(results, f)

with open('results_eval.pkl', 'rb') as f:
    results = pickle.load(f)

print(results)
#
# print("comet: ", np.mean([x['f1'][0] for x in bert_score_results]), "+-",
#       np.var([x['f1'][0] for x in bert_score_results]))
# print("lg: ", np.mean([x['f1'][1] for x in bert_score_results]), "+-", np.var([x['f1'][1] for x in bert_score_results]))
# print("md: ", np.mean([x['f1'][2] for x in bert_score_results]), "+-", np.var([x['f1'][2] for x in bert_score_results]))
# print("sd: ", np.mean([x['f1'][3] for x in bert_score_results]), "+-", np.var([x['f1'][3] for x in bert_score_results]))
