import argparse
import logging
import os
import random
import time
import json
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax

from torch.nn import CrossEntropyLoss
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from tqdm import tqdm
from models.examples_to_features import (
    examples_to_features_converters, InputExample,
    create_examples, get_dataloader_and_text_ids_with_sequence_ids,
    models, tokenizers, DataProcessor
)
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support
from torch.nn import CrossEntropyLoss


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_metrics(text_preds, text_labels, sequence_preds, sequence_labels, label2id):
    eval_seqience_labs = [
        label2id['sequence'][x] for x in label2id['sequence'] if x != '0'
    ]
    text_p, text_r, text_f1, _ = precision_recall_fscore_support(
        text_labels, text_preds,
        labels=[0, 1], average='weighted'
    )
    seq_p, seq_r, seq_f1, _ = precision_recall_fscore_support(
        sequence_labels, sequence_preds,
        labels=eval_seqience_labs, average="weighted"
    )
    result = {
        'text_f1': text_f1,
        'text_p': text_p,
        'text_r': text_r,
        'sequence_f1': seq_f1,
        'sequence_p': seq_p,
        'sequence_r': seq_r
    }
    return result


def evaluate(
        model, device, eval_dataloader, eval_text_labels_ids,
        eval_sequence_labels_ids, num_text_labels,
        num_sequence_labels, label2id,
        compute_scores=True, verbose=True
    ):
    model.eval()
    text_clf_weight = model.text_clf_weight
    sequence_clf_weight = model.sequence_clf_weight

    eval_loss = defaultdict(float)
    nb_eval_steps = 0
    preds = defaultdict(list)

    for batch in tqdm(
            eval_dataloader, total=len(eval_dataloader),
            desc='validating...'
        ):
        batch = tuple([elem.to(device) for elem in batch])
        input_ids, input_mask, segment_ids, \
            text_labels_ids, sequence_labels_ids = batch

        with torch.no_grad():
            output = model(
                input_ids=input_ids,
                token_type_ids=segment_ids,
                attention_mask=input_mask,
                text_labels=None,
                sequence_labels=None
            )

        sequence_logits, text_logits = output[:2]
        loss_fct = CrossEntropyLoss()
        tmp_text_eval_loss = loss_fct(
            text_logits.view(-1, num_text_labels),
            text_labels_ids.view(-1)
        )
        eval_loss['text'] += tmp_text_eval_loss.mean().item()

        loss_fct = CrossEntropyLoss(ignore_index=0)
        active_loss = input_mask.view(-1) == 1
        active_labels = sequence_labels_ids.view(-1)[active_loss]
        active_logits = sequence_logits.view(-1, num_sequence_labels)[active_loss]
        tmp_sequence_eval_loss = loss_fct(active_logits, active_labels)
        eval_loss['sequence'] += tmp_sequence_eval_loss.mean().item()

        eval_loss['weighted_loss'] = \
            text_clf_weight * eval_loss['text'] + sequence_clf_weight * eval_loss['sequence']

        nb_eval_steps += 1
        preds['text'].append(text_logits.detach().cpu().numpy())
        preds['sequence'].append(sequence_logits.detach().cpu().numpy())

    preds['text'] = np.concatenate(preds['text'], axis=0)
    preds['sequence'] = np.concatenate(preds['sequence'], axis=0)

    scores = {}
    for key in eval_loss:
        eval_loss[key] = eval_loss[key] / nb_eval_steps

    for key in preds:
        scores[key] = softmax(preds[key], axis=-1).max(axis=-1)
        preds[key] = preds[key].argmax(axis=-1)

    if compute_scores:
        result = compute_metrics(
            preds['text'], eval_text_labels_ids.numpy(),
            np.array([x for y in preds['sequence'] for x in y]),
            np.array([x for y in eval_sequence_labels_ids.numpy() for x in y]),
            label2id
        )
    else:
        result = {}

    for key in eval_loss:
        result[f'eval_loss_{key}'] = eval_loss[key]
    if verbose:
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
    return preds, result, scores


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()

    if args.gradient_accumulation_steps < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                args.gradient_accumulation_steps
            )
        )

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    if args.do_train:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.do_train:
        if os.path.exists(os.path.join(args.output_dir, 'train.log')):
            suffix = datetime.now().isoformat().replace('-', '_').replace(':', '_').split('.')[0].replace('T', '-')
            log_file = os.path.join(args.output_dir, 'train.log')
            os.system(f'cp {log_file} {log_file}_dump_at_{suffix}.log')
        logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, "train.log"), 'w'))
    else:
        logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, "eval.log"), 'w'))
    logger.info(args)
    logger.info("device: {}, n_gpu: {}".format(device, n_gpu))

    processor = DataProcessor(tag_format=args.tag_format)
    text_labels_list = processor.get_text_labels(args.data_dir, logger)
    sequence_labels_list = processor.get_sequence_labels(args.data_dir, logger)

    label2id = {
        'text': {
            label: i for i, label in enumerate(text_labels_list)
        },
        'sequence': {
            label: i for i, label in enumerate(sequence_labels_list, 1)
        }
     }

    id2label = {
        'text': {
            i: label for i, label in enumerate(text_labels_list)
        },
        'sequence': {
            i: label for i, label in enumerate(sequence_labels_list, 1)
        }
    }

    num_text_labels = len(text_labels_list)
    num_sequence_labels = len(sequence_labels_list) + 1

    do_lower_case = 'uncased' in args.model
    tokenizer = tokenizers[args.model].from_pretrained(args.model, do_lower_case=do_lower_case)
    convert_examples_to_features = examples_to_features_converters[args.model]

    eval_examples = processor.get_dev_examples(args.data_dir)
    eval_features = convert_examples_to_features(
        eval_examples, label2id, args.max_seq_length, tokenizer, logger
    )
    logger.info("***** Dev *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_dataloader, eval_text_labels_ids, eval_sequence_labels_ids = \
        get_dataloader_and_text_ids_with_sequence_ids(eval_features, args.eval_batch_size)

    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        train_features = convert_examples_to_features(
            train_examples, label2id,
            args.max_seq_length, tokenizer, logger
        )

        if args.train_mode == 'sorted' or args.train_mode == 'random_sorted':
            train_features = sorted(train_features, key=lambda f: np.sum(f.input_mask))
        else:
            random.shuffle(train_features)

        train_dataloader, _, _ = \
            get_dataloader_and_text_ids_with_sequence_ids(train_features, args.train_batch_size)
        train_batches = [batch for batch in train_dataloader]

        num_train_optimization_steps = \
            len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        warmup_steps = int(args.warmup_proportion * num_train_optimization_steps)

        logger.info("***** Training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        best_result = None
        eval_step = max(1, len(train_batches) // args.eval_per_epoch)
        lrs = [args.learning_rate] if args.learning_rate else \
            [1e-6, 2e-6, 3e-6, 5e-6, 1e-5, 2e-5, 3e-5, 5e-5]
        for lr in lrs:
            model = models[args.model].from_pretrained(
                args.model, cache_dir=str(PYTORCH_PRETRAINED_BERT_CACHE),
                num_text_labels=num_text_labels,
                num_sequence_labels=num_sequence_labels,
                sequence_clf_weight=args.sequence_clf_weight,
                text_clf_weight=args.text_clf_weight
            )
            model.to(device)

            if n_gpu > 1:
                model = torch.nn.DataParallel(model)

            param_optimizer = list(model.named_parameters())
            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {
                    'params': [
                        param for name, param in param_optimizer
                        if not any(nd in name for nd in no_decay)
                    ],
                    'weight_decay': float(args.weight_decay)
                },
                {
                    'params': [
                        param for name, param in param_optimizer
                        if any(nd in name for nd in no_decay)
                    ],
                    'weight_decay': 0.0
                }
            ]

            optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=lr
            )
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_train_optimization_steps
            )

            start_time = time.time()
            global_step = 0
            tr_loss = 0
            nb_tr_examples = 0
            nb_tr_steps = 0
            for epoch in range(1, 1 + int(args.num_train_epochs)):
                model.train()
                logger.info("Start epoch #{} (lr = {})...".format(epoch, lr))
                if args.train_mode == 'random' or args.train_mode == 'random_sorted':
                    random.shuffle(train_batches)
                    
                for step, batch in enumerate(tqdm(train_batches, total=len(train_batches), desc='fitting ... ')):
                    batch = tuple(t.to(device) for t in batch)
                    input_ids, input_mask, segment_ids, text_labels_ids, sequence_labels_ids = batch
                    loss = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask,
                                 text_labels=text_labels_ids,
                                 sequence_labels=sequence_labels_ids)

                    if n_gpu > 1:
                        loss = loss.mean()

                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    tr_loss += loss.item()
                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1

                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        global_step += 1

                    if args.do_validate and (step + 1) % eval_step == 0:
                        logger.info('Epoch: {}, Step: {} / {}, used_time = {:.2f}s, loss = {:.6f}'.format(
                                epoch, step + 1, len(train_batches),
                                time.time() - start_time, tr_loss / nb_tr_steps
                            )
                        )
                        save_model = False

                        preds, result, scores = evaluate(
                            model, device, eval_dataloader, eval_text_labels_ids,
                            eval_sequence_labels_ids,
                            num_text_labels, num_sequence_labels,
                            label2id
                        )
                        model.train()
                        result['global_step'] = global_step
                        result['epoch'] = epoch
                        result['learning_rate'] = lr
                        result['batch_size'] = args.train_batch_size
                        logger.info("First 20 predictions:")
                        for text_pred, text_label in zip(
                                preds['text'][:20],
                                eval_text_labels_ids.numpy()[:20]):
                            sign = u'\u2713' if text_pred == text_label else u'\u2718'
                            logger.info("pred = %s, label = %s %s" % (id2label['text'][text_pred],
                                                                      id2label['text'][text_label], sign))

                        if (best_result is None) or (result[args.eval_metric] > best_result[args.eval_metric]):
                            best_result = result
                            save_model = True
                            logger.info("!!! Best dev %s (lr=%s, epoch=%d): %.2f" %
                                        (args.eval_metric, str(lr), epoch, result[args.eval_metric] * 100.0)
                            )

                        if save_model:
                            model_to_save = model.module if hasattr(model, 'module') else model
                            output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
                            output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
                            torch.save(model_to_save.state_dict(), output_model_file)
                            model_to_save.config.to_json_file(output_config_file)
                            tokenizer.save_vocabulary(args.output_dir)
                            if best_result:
                                output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
                                with open(output_eval_file, "w") as writer:
                                    for key in sorted(result.keys()):
                                        writer.write("%s = %s\n" % (key, str(result[key])))
    if args.do_eval:
        test_file = os.path.join(args.data_dir, 'test.json') if args.test_file == '' else args.test_file
        eval_examples = processor.get_test_examples(test_file)

        eval_features = convert_examples_to_features(
            eval_examples, label2id, args.max_seq_length, tokenizer, logger
        )
        logger.info("***** Test *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)

        eval_dataloader, eval_text_labels_ids, eval_sequence_labels_ids = \
            get_dataloader_and_text_ids_with_sequence_ids(eval_features, args.eval_batch_size)

        model = models[args.model].from_pretrained(
            args.output_dir, num_sequence_labels=num_sequence_labels,
            num_relation_labels=num_relation_labels,
            text_clf_weight=args.text_clf_weight,
            sequence_clf_weight=args.sequence_clf_weight,
        )
        model.to(device)

        preds, result, scores = evaluate(
            model, device, eval_dataloader, eval_text_labels_ids,
            eval_sequence_labels_ids,
            num_text_labels, num_sequence_labels, label2id,
            compute_scores=False
        )

        aggregated_results = {}
        task = "sequence"
        aggregated_results[task] = [
            list(pred[orig_positions]) + [label2id[task]['0']] * (len(ex.tokens) - len(orig_positions))
            for pred, orig_positions, ex in zip(
                preds[task],
                eval_orig_positions_map,
                eval_examples
            )
        ]
        aggregated_results[f'{task}_scores'] = [
            list(score[orig_positions]) + [0.999] * (len(ex.sentence) - len(orig_positions))
            for score, orig_positions, ex in zip(
                scores[task],
                eval_orig_positions_map,
                eval_examples
            )
        ]

        prediction_results = {
            'idx': [
                ex.guid for ex in eval_examples
            ],
            'tokens': [
                 ' '.join(ex.tokens) for ex in eval_examples
            ],
            'sequence_labels': [
                ' '.join(ex.sequence_labels) for ex in eval_examples
            ],
            'text_label': [
                ex.text_label for ex in eval_examples
            ],
            'text_pred': [
                id2label['text'][x] for x in preds['text']
            ],
            'sequence_pred': [
                ' '.join([id2label['sequence'][x] for x in sent])
                for sent in aggregated_results['sequence']
            ],
            'sequence_scores': [
                ' '.join([str(score) for score in enumerate(sent)])
                for sent in aggregated_results['sequence_scores']
            ]
        }

        prediction_results = pd.DataFrame(prediction_results)
        prediction_results.to_csv(
            os.path.join(args.output_dir, f"{args.test_file.split('/')[-1]}_predictions.tsv"),
            sep='\t', index=False
        )
        with open(os.path.join(
                args.output_dir, f"{args.test_file.split('/')[-1]}_eval_results.txt"), "w") as f:

            for key in sorted(result.keys()):
                f.write("%s = %s\n" % (key, str(result[key])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", default='', type=str, required=False)
    parser.add_argument("--tag_format", default='bieo', type=str, required=False)
    parser.add_argument("--model", default=None, type=str, required=True)
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .json files for the task.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--eval_per_epoch", default=3, type=int,
                        help="How many times to do validation on dev set per epoch")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization.\n"
                             "Sequences longer than this will be truncated, and sequences shorter\n"
                             "than this will be padded.")

    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--train_mode", type=str, default='random_sorted',
                        choices=['random', 'sorted', 'random_sorted'])
    parser.add_argument("--do_validate", action='store_true', help="Whether to run validation on dev set.")

    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the test set.")
    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_metric", default="text_f1", type=str)
    parser.add_argument("--learning_rate", default=None, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=5.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup.\n"
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="maximal gradient norm")

    parser.add_argument("--text_clf_weight", default=1.0, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--sequence_clf_weight", default=1.0, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")

    parser.add_argument("--weight_decay", default=0.1, type=float,
                        help="weight_decay coefficient for regularization")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    arguments = parser.parse_args()
    main(arguments)
