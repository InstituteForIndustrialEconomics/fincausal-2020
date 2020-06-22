tag_fmt=$2
num_train_epochs=10
max_seq_length=128
train_batch_size=32
gradient_accumulation_steps=4
eval_per_epoch=16

for lr in 1e-4 5e-4 7e-4;
do
  for seq_weight in 1.0;
  do
    for text_weight in 0.0;
    do
      for wd in 0.1;
      do
        for drop in 0.3 0.5
        do
          CUDA_VISIBLE_DEVICES="$1" python run_fincausal.py \
            --model "bert-large-uncased" \
            --data_dir "data" \
            --do_train \
            --do_validate \
            --output_dir "tmp_bert_models/best_sequence_16_per_epoch/lr-$lr.text_weight-$text_weight-seq_weight-$seq_weight.wd-$wd.drop-$drop.tag_fmt-$tag_fmt" \
            --max_seq_length "$max_seq_length" \
            --train_batch_size "$train_batch_size" \
            --gradient_accumulation_steps "$gradient_accumulation_steps" \
            --eval_per_epoch "$eval_per_epoch" \
            --num_train_epochs $num_train_epochs \
            --weight_decay $wd \
            --dropout $drop \
            --text_clf_weight $text_weight \
            --sequence_clf_weight $seq_weight \
            --learning_rate $lr \
            --tag_format "$tag_fmt" \
            --eval_metric "sequence_weighted avg_f1-score" \
            --only_task_2;
          done
        done
    done
  done
done