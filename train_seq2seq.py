import sys
sys.dont_write_bytecode = True
import os
import numpy as np
import logging
from itertools import chain
from argparse import Namespace
from datasets import load_dataset, load_metric
from transformers import (
    AutoConfig, 
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
)
from transformers.trainer_utils import get_last_checkpoint
from DG_dataset import DGDataset

logger = logging.getLogger(__name__)



def main(args):
    data_args = Namespace(
        model_name_or_path=args.model_name_or_path,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        val_max_target_length=args.val_max_target_length,
        pad_to_max_length=args.pad_to_max_length,
        ignore_pad_token_for_loss=True,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        max_predict_samples=args.max_predict_samples,
        preprocessing_num_workers=args.preprocessing_num_workers,
        overwrite_cache=args.overwrite_cache,
        output_dir=args.output_dir,
        num_beams=args.num_beams,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=data_args.output_dir,
        do_train=args.do_train,
        do_eval=args.do_eval,
        do_predict=args.do_predict,
        evaluation_strategy="epoch",
        metric_for_best_model="eval_bleu",
        greater_is_better=True, # smaller eval loss is better
        save_total_limit=2, # save 2 checkpoints (best and last)
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        predict_with_generate=True, # generation task
    )

    padding = "max_length" if data_args.pad_to_max_length else False

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )

        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Blended Skill Talk
    bst_dataset = load_dataset("blended_skill_talk")

    # Tokenizer and model
    config = AutoConfig.from_pretrained(data_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(data_args.model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(data_args.model_name_or_path, config=config)

    # Add special tokens
    # Define new special tokens: <PS>, <CTX>, <SEP>
    tokenizer.add_tokens(['<PS>'], special_tokens=True) ## this line is updated
    tokenizer.add_tokens(['<CTX>'], special_tokens=True) ## this line is updated
    tokenizer.add_tokens(['<SEP>'], special_tokens=True) ## this line is updated
    model.resize_token_embeddings(len(tokenizer))

    # Data processing
    dg = DGDataset(
        dataset=args.dataset,
        task='seq2seq',
        tokenizer=tokenizer,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        padding=padding,
        ignore_pad_token_for_loss=data_args.ignore_pad_token_for_loss,
        preprocessing_num_workers=args.preprocessing_num_workers,
        overwrite_cache=args.overwrite_cache,
    )

    # Tokenize train, eval, test dataset
    if training_args.do_train:
        train_dataset = bst_dataset['train']
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        train_dataset = dg.preprocess(train_dataset)
        print("train dataset: ", train_dataset)
    
    if training_args.do_eval:
        eval_dataset = bst_dataset['validation']
        max_target_length = data_args.val_max_target_length
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
        eval_dataset = dg.preprocess(eval_dataset)
        print("validation dataset: ", eval_dataset)

    if training_args.do_predict:
        predict_dataset = bst_dataset['test']
        max_target_length = data_args.val_max_target_length
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))
        predict_dataset = dg.preprocess(predict_dataset)
        print("test dataset: ", predict_dataset)


    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )

    # Metric
    metric = load_metric("sacrebleu")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]
        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": result["score"]}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()


    # Evaluation
    results = {}
    max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(max_length=max_length, num_beams=num_beams, metric_key_prefix="eval")
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")
        predict_results = trainer.predict(
            predict_dataset, metric_key_prefix="predict", max_length=max_length, num_beams=num_beams
        )
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = tokenizer.batch_decode(
                    predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                predictions = [pred.strip() for pred in predictions]
                output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
                with open(output_prediction_file, "w") as writer:
                    writer.write("\n".join(predictions))

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='t5-small', 
                        help='The model checkpoint for weights initialization.')
    parser.add_argument("--dataset", "-d", type=str, default="blended_skill_talk", 
                        choices=[
                            "blended_skill_talk",
                            "conv_ai_2",
                            "empathetic_dialogues",
                            "AlekseyKorshuk/persona-chat",
                        ], 
                        help='The dataset to use for training.')
    parser.add_argument('--output_dir', type=str, default='results/t5',
                        help='The output directory where the model predictions and checkpoints will be written.')
    parser.add_argument('--num_train_epochs', type=int, default=50,
                        help='Total number of training epochs to perform.')
    parser.add_argument('--per_device_train_batch_size', type=int, default=10,
                        help='Batch size per GPU/CPU for training.')
    parser.add_argument('--per_device_eval_batch_size', type=int, default=20,
                        help='Batch size per GPU/CPU for evaluation.')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=20,
                        help='Number of updates steps to accumulate before performing a backward/update pass.')
    parser.add_argument('--do_train', action='store_true',
                        help='Whether to run training.')
    parser.add_argument('--do_eval', action='store_true',
                        help='Whether to run eval on the dev set.')
    parser.add_argument('--do_predict', action='store_true',
                        help='Whether to run predictions on the test set.')
    parser.add_argument('--max_train_samples', type=int, default=None,
                        help='For debugging purposes or quicker training, truncate the number of training examples to this.')
    parser.add_argument('--max_eval_samples', type=int, default=None,
                        help='For debugging purposes or quicker training, truncate the number of evaluation examples to this.')
    parser.add_argument('--max_predict_samples', type=int, default=None,
                        help='For debugging purposes or quicker training, truncate the number of prediction examples to this.')
    parser.add_argument('--max_source_length', type=int, default=512,
                        help='The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.')
    parser.add_argument('--max_target_length', type=int, default=512,
                        help='The maximum total sequence length for target text after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.')
    parser.add_argument('--val_max_target_length', type=int, default=512,
                        help='The maximum total sequence length for validation target text after tokenization.Sequences longer than this will be truncated, sequences shorter will be padded.')
    parser.add_argument('--pad_to_max_length', action='store_true',
                        help='Whether to pad all samples to model maximum sentence length.')
    parser.add_argument('--num_beams', type=int, default=4,
                        help='Number of beams to use for evaluation. This argument will be passed to ``model.generate``, which is used during ``evaluate`` and ``predict``.')
    parser.add_argument('--overwrite_cache', action='store_true',
                        help='Overwrite the cached training and evaluation sets')
    parser.add_argument('--preprocessing_num_workers', type=int, default=None,
                        help='The number of processes to use for the preprocessing.')

    args = parser.parse_args()

    main(args)

