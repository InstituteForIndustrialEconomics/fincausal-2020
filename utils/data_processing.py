import fire
import pandas as pd


def csv_reader(
        path: str,
        sep: str = '; '
    ):
    result = pd.read_csv(path, sep=sep, engine="python")

    return result


def merge_tasks(
        path_to_task_1: str,
        path_to_task_2: str,
        sep: str = ','
    ):
    task_1 = csv_reader(path_to_task_1, sep=sep)
    task_2 = csv_reader(path_to_task_2, sep=sep)
    return task_1, task_2


if __name__ == "__main__":
    fire.Fire(merge_tasks)
