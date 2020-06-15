import pandas as pd
from typing import List
import json
from utils.global_variables import (task_1_texts_with_label_collisions,
                                    task_2_texts_with_fact_collisions)
import nltk
from collections import defaultdict


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
        else:
            new_df["gold"].append("1")
            causes = set(zip(cur_df.cause_start, cur_df.cause_end))
            effects = set(zip(cur_df.effect_start, cur_df.effect_end))

            if not (len(causes) == 1 or len(effects) == 1):
                new_df["causes"].append(
                    [(cur_df.cause_start.values[0], cur_df.cause_end.values[0])]
                )
                new_df["effects"].append(
                    [(cur_df.effect_start.values[0], cur_df.effect_end.values[0])]
                )
            else:
                new_df["causes"].append(list(causes))
                new_df["effects"].append(list(effects))

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


def tokenize_dataset(
        dataset: pd.DataFrame
    ):
    df = dataset.copy()
    df.loc[:, 'tokenized_text'] = df.text.apply(nltk.word_tokenize)

    return df
