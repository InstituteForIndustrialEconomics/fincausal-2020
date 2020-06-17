#Fincausal 2020 shared task

This repo contains a code for training proposed models for Fincausal 2020 shared
task. To run the code you need python 3.

To create fincausal_env environment and install requirements into it run following commands from the root of the repo:
```bash
python3 -m venv fincausal_env
. fincausal_env/bin/activate
pip install -r requirements.txt
deactivate
```

To train the model run:
```bash
python run_fincausal.py \
    --model bert-large-uncased \
    --data_dir data \
    --do_train \
    --do_validate \
    --output_dir directory_to_save_the_model \
    --max_seq_length 128 \
    --eval_per_epoch 4
```

To make predictions on the test set run:
```bash
python run_fincausal.py \
    --model bert-large-uncased \
    --data_dir data \
    --do_eval \
    --output_dir directory_with_saved_model \
    --max_seq_length 128
```

To show all hyperparameters of the model:
```bash
python run_fincausal.py -h
```