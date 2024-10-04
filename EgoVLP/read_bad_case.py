import json

import pandas as pd
from tqdm import tqdm

verbs_in_actions = []
mapping = {}
sv = {}

with open(
    "/fs/fast/base_path/annotations/egovlpv3/egomcq.json", "r", encoding="utf-8"
) as f:
    dtt = json.load(f)
with open("./bad_case_list.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        if not int(data["pred"]) == int(data["answer"]):
            import pdb

            pdb.set_trace()
            sv[str(len(sv))] = data
print(len(sv))
with open("/fs/fast/base_path/annotations/egovlpv3/egomcq_bad_case.json", "w") as f:
    json.dump(sv, f)
