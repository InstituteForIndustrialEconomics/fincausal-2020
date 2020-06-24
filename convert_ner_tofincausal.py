import json

with open("data/test-task-2.json") as f:
    test = json.load(f)

with open("test_predictions_task2bert2.txt") as f:
    predictions = f.readlines()
predictions = [p.strip().split() for p in predictions]

sents = []
sent = []
for p in predictions:
    if not p:
        sents.append(sent)
        sent = []
    else:
        sent.append(p)


def get_ners(test_sent_i, test_sent):
    pred_sent = sents[test_sent_i]
    pred_token = pred_sent[0][0]
    if not test_sent["tokens"][0] == pred_token:
        errors += 1
        return dict()
    total_len = 0
    ners = dict()
    for word_i, (word, token) in enumerate(pred_sent):
        text = test_sent["text"][total_len:]
        if token != "0":
            ner_type = token[-1]
            word_index = text.find(word) + total_len
            if ner_type not in ners:
                ners[ner_type] = dict()
                ners[ner_type]["start"] = word_index
            else:
                ners[ner_type]["end"] = word_index + len(word) + 1
        total_len += len(word)
    return ners


with open("preds_fincausal_format.csv", "w") as f:
    f.write("Index; Text; Cause; Effect; Offset_Sentence2; Offset_Sentence3\n")
errors = 0

df = pd.read_csv("data/task2_sample_submission.csv", sep=";")
causes = []
effects = []
for test_sent_i, test_sent in enumerate(test):
    try:
        ners = get_ners(test_sent_i, test_sent)
    except IndexError:
        print(test_sent_i)
        ners = dict()
    sent_ind = test_sent["idx"]
    text = test_sent["text"]
    try:
        cause = test_sent["text"][ners["C"]["start"] : ners["C"]["end"]]
        effect = test_sent["text"][ners["E"]["start"] : ners["E"]["end"]]
    except KeyError as ex:
        cause = ""
        effect = ""
    causes.append(cause)
    effects.append(effect)

df[" Cause"] = causes
df[" Effect"] = effects

df.to_csv("task2.csv", sep=";", index=None)
