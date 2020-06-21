import pandas as pd
from typing import List
import json
from utils.global_variables import (task_1_texts_with_label_collisions,
                                    task_2_texts_with_fact_collisions)
import nltk
from collections import defaultdict
from sklearn.model_selection import train_test_split


def csv_reader(
        path: str,
        sep: str = '; ',
        index_name: str = "idx"
    ):
    result = pd.read_csv(
        path,
        sep=sep,
        engine="python",
        dtype={
            "Index": str, "Text": str, "Gold": str,
            "Cause": str, "Effect": str, "Offset_Sentence2": float,
            "Offset_Sentence3": float, "Cause_Start": int,
            "Cause_End": int, "Effect_Start": int, "Effect_End": int,
            "Sentence": str
        }
    )
    result.columns = [column.lower() for column in result.columns]
    result = result.rename(mapper={"index": index_name}, axis=1)
    result = result[~result.text.isna()]

    return result


def merge_tasks(
        paths_to_task_1: List[str],
        paths_to_task_2: List[str],
        sep: str = '; ',
    ):
    task_1 = concat_dataset_parts(paths_to_task_1, sep=sep, clean=True)
    task_2 = concat_dataset_parts(paths_to_task_2, sep=sep, clean=True)

    new_df = defaultdict(list)
    for i, row in enumerate(task_1.itertuples(), 1):
        text = row.text
        cur_df = task_2[task_2.text == text]

        new_df["idx"].append(f"{row.idx}-{i}")
        new_df["text"].append(text)

        if len(cur_df) == 0:
            new_df["gold"].append("0")
            new_df["causes"].append([])
            new_df["effects"].append([])
            new_df["task_id"].append("1")
        else:
            new_df["gold"].append("1")
            causes = set(cur_df.cause)
            effects = set(cur_df.effect)

            if not (len(causes) == 1 or len(effects) == 1):
                new_df["causes"].append(
                    [cur_df.cause.values[0]]
                )
                new_df["effects"].append(
                    [cur_df.effect.values[0]]
                )
            else:
                new_df["causes"].append(list(causes))
                new_df["effects"].append(list(effects))

            new_df["task_id"].append("2")

    new_df = pd.DataFrame(new_df)
    return new_df


def clean_dataset(
        dataset: pd.DataFrame,
        task_id: int = 1
    ):
    data = dataset.copy()
    if task_id == 1:
        for text, label in task_1_texts_with_label_collisions:
            data.loc[data.text == text, 'gold'] = label
        data = data.drop_duplicates(subset=["text"])
    elif task_id == 2:
        data = data.drop_duplicates(subset=["text", "cause_start", "cause_end", "effect_start", "effect_end"])
        for text in task_2_texts_with_fact_collisions:
            data = data[data.text != text]
    return data


def concat_dataset_parts(
        paths_to_parts: List[str],
        sep: str = '; ',
        task_id: int = 1,
        clean: bool = True
    ):
    data = csv_reader(paths_to_parts[0], sep=sep)

    for path in paths_to_parts[1:]:
        data = data.append(csv_reader(path, sep=sep), ignore_index=True)

    if clean:
        data = clean_dataset(dataset=data, task_id=task_id)

    return data


def create_merged_train_examples(
        paths_to_task_1: List[str],
        paths_to_task_2: List[str],
        sep: str = "; ",
        tag_format: str = "bio"
    ):
    merged_df = merge_tasks(paths_to_task_1, paths_to_task_2, sep)
    examples = []
    not_found_causes_num = 0
    not_found_effects_num = 0

    for row in merged_df.itertuples():
        cause_not_found = True
        effect_not_found = True

        tokens = nltk.word_tokenize(row.text)
        causes = [nltk.word_tokenize(cause) for cause in row.causes]
        effects = [nltk.word_tokenize(effect) for effect in row.effects]
        labels = ['0'] * len(tokens)

        for cause in causes:
            cause_positions = get_sublist_positions(tokens, cause)
            if len(cause_positions) > 0:
                cause_not_found = False
                filling_values = get_formatted_labels("C", len(cause), tag_format)
                for i, lab_pos in enumerate(cause_positions):
                    labels[lab_pos] = filling_values[i]
            else:
                print(tokens, cause)

        for effect in effects:
            effect_positions = get_sublist_positions(tokens, effect)
            if len(effect_positions) > 0:
                effect_not_found = False
                filling_values = get_formatted_labels("E", len(effect), tag_format)
                for i, lab_pos in enumerate(effect_positions):
                    labels[lab_pos] = filling_values[i]

        not_found_causes_num += 1 if cause_not_found and len(causes) else 0
        not_found_effects_num += 1 if effect_not_found and len(effects) else 0

        if (cause_not_found or effect_not_found) and row.gold != "0":
            continue

        example = {
            "idx": row.idx,
            "text": row.text,
            "tokens": tokens,
            "text_label": row.gold,
            "sequence_labels": labels,
            "task_id": row.task_id
        }
        examples.append(example)

    print(f"not found causes: {not_found_causes_num}")
    print(f"not found effects: {not_found_effects_num}")
    print(f"total number of examples: {len(merged_df)}")

    return examples


def create_test_examples(
        path_to_test: str,
        sep: str = "; "
    ):
    """

    :param path_to_test: path to test csv file
    :param sep: columns separator in csv file
    :return: list of examples, where each example is a dictionary
    """
    test = csv_reader(path_to_test, sep=sep)
    examples = []
    for row in test.itertuples():
        tokens = nltk.word_tokenize(row.text)
        example = {
            "idx": row.idx,
            "text": row.text,
            "tokens": tokens,
            "text_label": '0',
            "sequence_labels": ['0'] * len(tokens),
            "task_id": "1+2"
        }
        examples.append(example)
    return examples


def get_formatted_labels(
        label: str,
        labels_len: int,
        tag_format: str = "bio"
    ):
    if tag_format == "bio":
        labels = [f"I-{label}"] * labels_len
        labels[0] = f"B-{label}"

    elif tag_format == "se":
        labels = ["0"] * labels_len
        labels[0] = f"S-{label}"
        labels[-1] = f"E-{label}"

    elif tag_format == "bieo":
        labels = [f"I-{label}"] * labels_len
        labels[-1] = f"E-{label}"
        labels[0] = f"B-{label}"

    else:
        raise ValueError

    return labels


def get_sublist_positions(
        whole_list: list,
        sublist: list
    ):
    result = []
    for i in range(len(whole_list) - len(sublist) + 1):
        if sublist == whole_list[i: i + len(sublist)]:
            result = list(range(i, len(sublist) + i))
    return result


def get_train_and_validation_examples(
        paths_to_task_1: List[str],
        paths_to_task_2: List[str],
        sep: str = "; ",
        tag_format: str = "bio",
        validation_ratio: int = 0.25
    ):
    examples = create_merged_train_examples(
        paths_to_task_1,
        paths_to_task_2,
        sep,
        tag_format
    )

    train, validation = train_test_split(
        examples, test_size=validation_ratio,
        random_state=2020,
        stratify=[ex["text_label"] for ex in examples]
    )
    return train, validation


def write_task_1_predictions(
        task_1_predictions_csv: str,
        output_file: str
    ):
    predictions = pd.read_csv(task_1_predictions_csv, sep='\t')
    with open(output_file, 'w') as f:
        print("Index; Text; Prediction", file=f)
        for row in predictions.itertuples():
            idx = row.idx.split("-")[1]
            text = row.text
            pred = str(row.text_pred)
            print("; ".join([idx, text, pred]), file=f)


def write_task_2_predictions(
        task_2_predictions_csv: str,
        output_file: str
    ):
    # TODO: implement span constructing strategy
    predictions = pd.read_csv(task_2_predictions_csv, sep='\t')
    with open(output_file, 'w') as f:
        print("Index; Text; Cause; Effect", file=f)
        for row in predictions.itertuples():
            idx = row.idx.split("-")[1]
            text = row.text
            pred = row.text_pred
            print("; ".join([idx, text, pred]), file=f)


