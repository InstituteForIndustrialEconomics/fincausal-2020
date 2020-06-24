import pandas as pd
import razdel


# practice = pd.read_csv("data/practice-task-2.csv", sep=";")
# trial = pd.read_csv("data/trial-task-2.csv", sep=";")
# practice.columns = [c.strip() for c in practice.columns]
# trial.columns = [c.strip() for c in trial.columns]

with open("data/bio_train.json") as f:
    train = json.load(f)
with open("data/bio_valid.json") as f:
    dev = json.load(f)
with open("data/test-task-2.json") as f:
    test = json.load(f)
for dataset_name, dataset in (("train", train), ("dev", dev), ("test", test)):
    filename = f"data/bio_{dataset_name}_task2.csv"
    with open(filename, "w") as f:
        f.write("")
    f = open(, "a")
    for sent in dataset:
        for token_i, token in enumerate(sent["tokens"]):
            label = sent["sequence_labels"][token_i]
            f.write(f"{token} {label}\n")
        f.write("\n")
    f.close()