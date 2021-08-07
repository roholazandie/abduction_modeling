from pathlib import Path
import requests
from jinja2 import Template

BASE_DIR = Path(__file__).resolve().parent
API_BASE_URL = "http://127.0.0.1:8000"

if __name__ == '__main__':
    with open(BASE_DIR.joinpath('htmls', 'question.html'), 'r') as fh:
        question_html = fh.read()

    hit_type_params = {'title': 'Can you identify if a model generated common sense reasoning make sense?',
                       'keyword': 'Abductive Reasoning',
                       'description': 'We would like you to evaluate how accurately a model is able to generate '
                                      'a hypothesis of an event that can reasonably occur between two observations'}

    hit_type_resp = requests.get(f"{API_BASE_URL}/heman/v1/create_hit_type/", hit_type_params).json()
    print(hit_type_resp)

    with open(BASE_DIR.joinpath('results.csv'), 'r') as fh:
        next(fh)  # skipping the header
        for idx, line in enumerate(fh):
            if len(line.split(',')) == 4:
                ob1, ob2, generated_hypotheses, actual_hypothesis = line.split(',')
                if len(generated_hypotheses.split('.')) == 3:
                    gh1, gh2, gh3 = generated_hypotheses.split('.')
                else:
                    print(generated_hypotheses)
                    continue
            else:
                print(line)
                continue

            t = Template(question_html)

            rendered = t.render({'observation_1': ob1,
                                 'observation_2': ob2,
                                 'actual_hypothesis': actual_hypothesis,
                                 'generated_hypothesis_1': gh1,
                                 'generated_hypothesis_2': gh2,
                                 'generated_hypothesis_3': gh3})

            hit_params = {'hit_type_id': hit_type_resp['HITTypeId'],
                          'question': rendered}
            try:
                hit_resp = requests.get(f"{API_BASE_URL}/heman/v1/create_hit/", hit_params).json()
            except:
                print(f"hit failed for {idx}")
            print(idx)








