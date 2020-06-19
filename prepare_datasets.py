from utils.data_processing import *
import fire
import json
import os


def save_data_splits(tag_format: str, overwrite_test_set: bool = False):
    """
    saves splitted data in specified format in data directory
    :param overwrite_test_set: whether to overwrite existing test set
    :param tag_format: sequence labeling format, one of [bio, bieo, se]
    :return: None
    """
    train, val = get_train_and_validation_examples(
        [
            'data/trial-task-1.csv',
            'data/practice-task-1.csv'
        ],
        [
            'data/trial-task-2.csv',
            'data/practice-task-2.csv'
        ],
        sep='; ',
        tag_format=tag_format,
        validation_ratio=0.2
    )
    json.dump(train, open(f"data/{tag_format}_train.json", "w"))
    json.dump(val, open(f"data/{tag_format}_valid.json", "w"))

    if not os.path.exists('data/test.json') or overwrite_test_set:
        test_examples = create_test_examples("data/test-task-1.csv")
        json.dump(test_examples, open('data/test-task-1.json', 'w'))
        test_examples = create_test_examples("data/test-task-2.csv")
        json.dump(test_examples, open('data/test-task-2.json', 'w'))


if __name__ == "__main__":
    fire.Fire(save_data_splits)