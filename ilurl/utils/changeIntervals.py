import json

flowFile = "../../data/networks/2_1/flow.json"
CHANGE = 0.5

with open(flowFile) as f:
  data = json.load(f)

for item in data:
  item["interval"] = int(item["interval"] * CHANGE)

with open(flowFile, 'w') as json_file:
  json.dump(data, json_file)

