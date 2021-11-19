import datasets
import pickle
import numpy as np

# bleu = datasets.load_metric("bleu")
# bertscore = datasets.load_metric("bertscore")
# bleurt = datasets.load_metric("bleurt")
# ter = datasets.load_metric("ter")
# meteor = datasets.load_metric("meteor")
# rouge = datasets.load_metric("rouge")
#
# results = {}
#
# with open("../data/results/evalset_all_results.csv") as file_reader:
#     for i, line in enumerate(file_reader):
#         if i == 0:
#             continue
#
#         items = line.split(',')
#
#         story_id = items[0]
#         real_hypothesis = items[3]
#         comet_model = items[4].strip("'").strip("\"").strip()
#         abduction_gpt_lg_model = items[5].strip("'").strip("\"").strip()
#         abduction_gpt_md_model = items[6].strip("'").strip("\"").strip()
#         abduction_gpt_sm_model = items[7].strip("'").strip("\"").split('.')[0].strip()
#         if real_hypothesis.strip() == "" or comet_model.strip() == "" or abduction_gpt_sm_model.strip() == "" or abduction_gpt_md_model.strip() == "" or abduction_gpt_sm_model.strip() == "":
#             continue
#
#         if story_id in results:
#             results[story_id]["hypotheses"].append(real_hypothesis)
#         else:
#             results[story_id] = {"obs1": items[1],
#                                  "obs2": items[2],
#                                  "hypotheses": [real_hypothesis],
#                                  "comet": comet_model,
#                                  "lg": abduction_gpt_lg_model,
#                                  "md": abduction_gpt_md_model,
#                                  "sm": abduction_gpt_sm_model
#                                  }
#
# ############bleu
# comet_blue = bleu.compute(predictions=[results[key]['comet'].split() for key in results],
#                           references=[[x.split() for x in results[key]['hypotheses']] for key in results])
#
# lg_bleu = bleu.compute(predictions=[results[key]['lg'].split() for key in results],
#                        references=[[x.split() for x in results[key]['hypotheses']] for key in results])
#
# md_bleu = bleu.compute(predictions=[results[key]['md'].split() for key in results],
#                        references=[[x.split() for x in results[key]['hypotheses']] for key in results])
#
# sm_bleu = bleu.compute(predictions=[results[key]['sm'].split() for key in results],
#                        references=[[x.split() for x in results[key]['hypotheses']] for key in results])
#
# #######################bert
#
# bert_score_dict = bertscore.compute(predictions=[results[key]['comet'] for key in results],
#                                     references=[results[key]['hypotheses'][0] for key in results],
#                                     lang="en")
#
# comet_bert = {"precision": np.mean(bert_score_dict['precision']),
#               "recall": np.mean(bert_score_dict['recall']),
#               "f1": np.mean(bert_score_dict['f1'])}
#
# bert_score_dict = bertscore.compute(predictions=[results[key]['lg'] for key in results],
#                                     references=[results[key]['hypotheses'][0] for key in results],
#                                     lang="en")
#
# lg_bert = {"precision": np.mean(bert_score_dict['precision']),
#            "recall": np.mean(bert_score_dict['recall']),
#            "f1": np.mean(bert_score_dict['f1'])}
#
# bert_score_dict = bertscore.compute(predictions=[results[key]['md'] for key in results],
#                                     references=[results[key]['hypotheses'][0] for key in results],
#                                     lang="en")
#
# md_bert = {"precision": np.mean(bert_score_dict['precision']),
#            "recall": np.mean(bert_score_dict['recall']),
#            "f1": np.mean(bert_score_dict['f1'])}
#
# bert_score_dict = bertscore.compute(predictions=[results[key]['sm'] for key in results],
#                                     references=[results[key]['hypotheses'][0] for key in results],
#                                     lang="en")
#
# sm_bert = {"precision": np.mean(bert_score_dict['precision']),
#            "recall": np.mean(bert_score_dict['recall']),
#            "f1": np.mean(bert_score_dict['f1'])}
#
# ###############bluert
#
# bluert_score_dict = bleurt.compute(predictions=[results[key]['comet'] for key in results],
#                                    references=[results[key]['hypotheses'][0] for key in results])
# comet_bleurt = np.mean(bluert_score_dict['scores'])
#
# bluert_score_dict = bleurt.compute(predictions=[results[key]['lg'] for key in results],
#                                    references=[results[key]['hypotheses'][0] for key in results])
# lg_bleurt = np.mean(bluert_score_dict['scores'])
#
# bluert_score_dict = bleurt.compute(predictions=[results[key]['md'] for key in results],
#                                    references=[results[key]['hypotheses'][0] for key in results])
# md_bleurt = np.mean(bluert_score_dict['scores'])
#
# bluert_score_dict = bleurt.compute(predictions=[results[key]['sm'] for key in results],
#                                    references=[results[key]['hypotheses'][0] for key in results])
# sm_bleurt = np.mean(bluert_score_dict['scores'])
#
# ##########ter
# comet_ter = ter.compute(predictions=[results[key]['comet'] for key in results],
#                         references=[[results[key]['hypotheses'][0]] for key in results])
#
# lg_ter = ter.compute(predictions=[results[key]['lg'] for key in results],
#                      references=[[results[key]['hypotheses'][0]] for key in results])
#
# md_ter = ter.compute(predictions=[results[key]['md'] for key in results],
#                      references=[[results[key]['hypotheses'][0]] for key in results])
#
# sm_ter = ter.compute(predictions=[results[key]['sm'] for key in results],
#                      references=[[results[key]['hypotheses'][0]] for key in results])
#
# ########meteor
# meteor_score_dict = meteor.compute(predictions=[results[key]['comet'] for key in results],
#                                    references=[results[key]['hypotheses'][0] for key in results])
# comet_meteor = meteor_score_dict['meteor']
#
# meteor_score_dict = meteor.compute(predictions=[results[key]['lg'] for key in results],
#                                    references=[results[key]['hypotheses'][0] for key in results])
# lg_meteor = meteor_score_dict['meteor']
#
# meteor_score_dict = meteor.compute(predictions=[results[key]['md'] for key in results],
#                                    references=[results[key]['hypotheses'][0] for key in results])
# md_meteor = meteor_score_dict['meteor']
#
# meteor_score_dict = meteor.compute(predictions=[results[key]['sm'] for key in results],
#                                    references=[results[key]['hypotheses'][0] for key in results])
# sm_meteor = meteor_score_dict['meteor']
#
# #########rouge
# rouge_score_dict = rouge.compute(predictions=[results[key]['comet'] for key in results],
#                                  references=[results[key]['hypotheses'][0] for key in results])
# comet_rouge = rouge_score_dict
#
# rouge_score_dict = rouge.compute(predictions=[results[key]['lg'] for key in results],
#                                  references=[results[key]['hypotheses'][0] for key in results])
# lg_rouge = rouge_score_dict
#
# rouge_score_dict = rouge.compute(predictions=[results[key]['md'] for key in results],
#                                  references=[results[key]['hypotheses'][0] for key in results])
# md_rouge = rouge_score_dict
#
# rouge_score_dict = rouge.compute(predictions=[results[key]['sm'] for key in results],
#                                  references=[results[key]['hypotheses'][0] for key in results])
# sm_rouge = rouge_score_dict
#
# results = {"bertscore": {"comet": comet_bert, "lg": lg_bert, "md": md_bert, "sm": sm_bert},
#            "bleurt": {"comet": comet_bleurt, "lg": lg_bleurt, "md": md_bleurt, "sm": sm_bleurt},
#            "bleu": {"comet": comet_blue, "lg": lg_bleu, "md": md_bleu, "sm": sm_bleu},
#            "ter": {"comet": comet_ter, "lg": lg_ter, "md": md_ter, "sm": sm_ter},
#            "meteor": {"comet": comet_meteor, "lg": lg_meteor, "md": md_meteor, "sm": sm_meteor},
#            "rouge": {"comet": comet_rouge, "lg": lg_rouge, "md": md_rouge, "sm": sm_rouge},
#            }
#
# # print(bert_score_results)
# with open('results_automatic_eval.pkl', 'wb') as f:
#     pickle.dump(results, f)


with open('results_automatic_eval.pkl', 'rb') as f:
    results = pickle.load(f)

print(results)