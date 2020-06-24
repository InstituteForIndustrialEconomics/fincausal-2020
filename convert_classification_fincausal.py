import json

with open("data/test-task-1.json") as f:
    test = json.load(f)

with open("task_1_roberta_test_predictions.txt") as f:
    predictions = f.readlines()
predictions = [p.strip() for p in predictions]

df = pd.read_csv("data/task1_sample_submission.csv", sep=";")

df.to_csv("task2.csv", sep=";", index=None)
