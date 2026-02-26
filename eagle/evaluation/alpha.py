import json
import numpy as np

json_files=[
    "/home/lwtao/CalibrationEAGLE/mt_bench/baseline-temperature-0.0.jsonl",
    "/home/lwtao/CalibrationEAGLE/mt_bench/calibrated-temperature-0.0.jsonl",
]


for jsonl_file in json_files:
    data=[]
    with open(jsonl_file, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line)
            data.append(json_obj)

    alphas=[]
    for datapoint in data:
        qid=datapoint["question_id"]
        answer=datapoint["choices"][0]['turns']
        tokens=sum(datapoint["choices"][0]['new_tokens'])
        ids = sum(datapoint["choices"][0]['idxs'])
        alpha=datapoint["choices"][0]['alpha']
        alpha_num = datapoint["choices"][0]['alpha']    
        alphas.append(alpha)

    ar=np.array(alphas)
    print(np.mean(ar))