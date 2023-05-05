import sys
sys.dont_write_bytecode = True
import os
import math
import logging
from argparse import Namespace
import evaluate
from datasets import load_dataset
from transformers import (
    AutoConfig, 
    AutoTokenizer, 
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from DG_dataset import DGDataset
logger = logging.getLogger(__name__)



def main(args):
    data_args = Namespace(
        model_name_or_path=args.model_name_or_path,
        max_length=args.max_length,
        pad_to_max_length=args.pad_to_max_length,
        ignore_pad_token_for_loss=True,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        max_predict_samples=args.max_predict_samples,
        preprocessing_num_workers=args.preprocessing_num_workers,
        overwrite_cache=args.overwrite_cache,
        output_dir=args.output_dir,
        num_beams=args.num_beams,
        block_size=args.block_size,
    )

    training_args = TrainingArguments(
        output_dir=data_args.output_dir,
        do_train=args.do_train,
        do_eval=args.do_eval,
        do_predict=args.do_predict,
        seed=args.seed,
        evaluation_strategy="epoch",
        metric_for_best_model="eval_accuracy",
        greater_is_better=True, # smaller eval loss is better
        save_total_limit=2, # save 2 checkpoints (best and last)
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
    )

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
    
     # Set seed before initializing model.
    set_seed(training_args.seed)

    # Blended Skill Talk
    all_datasets = load_dataset(args.dataset)

    # Tokenizer and model
    config = AutoConfig.from_pretrained(data_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(data_args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(data_args.model_name_or_path, config=config)

    max_length = data_args.max_length
    padding = "max_length" if data_args.pad_to_max_length else False
    print("max length: {}, model max length: {}".format(max_length, tokenizer.model_max_length))

    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --block_size xxx."
            )
            block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    # Add special tokens
    tokenizer.add_special_tokens({'pad_token': '<PAD>'})
    tokenizer.add_special_tokens({'mask_token': '<MASK>'})
    model.resize_token_embeddings(len(tokenizer))

    # Data processing
    dg = DGDataset(
        dataset=args.dataset,
        task='clm',
        tokenizer=tokenizer,
        max_source_length=max_length,
        max_target_length=max_length,
        padding=padding,
        ignore_pad_token_for_loss=data_args.ignore_pad_token_for_loss,
        preprocessing_num_workers=args.preprocessing_num_workers,
        overwrite_cache=args.overwrite_cache,
    )

    # Tokenize train, eval, test dataset
    if training_args.do_train:
        train_dataset = all_datasets['train']
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        train_dataset = dg.preprocess(train_dataset)
        print("train dataset: ", train_dataset)
    
    if training_args.do_eval:
        eval_dataset = all_datasets['validation']
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

        metric = evaluate.load("accuracy")
        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics but we need to shift the labels
            labels = labels[:, 1:].reshape(-1)
            preds = preds[:, :-1].reshape(-1)
            return metric.compute(predictions=preds, references=labels)


        eval_dataset = dg.preprocess(eval_dataset)
        print("validation dataset: ", eval_dataset)

    if training_args.do_predict and 'test' in all_datasets:
        predict_dataset = all_datasets['test']
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))
        predict_dataset = dg.preprocess(predict_dataset)
        print("test dataset: ", predict_dataset)

    # Data collator
    data_collator = default_data_collator

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.do_eval else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if training_args.do_eval else None,
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
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='random seed for initialization')
    parser.add_argument('--model_name_or_path', 
                        type=str, 
                        default='microsoft/DialoGPT-small', 
                        choices=['microsoft/DialoGPT-small', 'gpt2'],
                        help='The model checkpoint for weights initialization.')
    parser.add_argument("--dataset", "-d", type=str, 
                        default="blended_skill_talk", 
                        choices=[
                            "blended_skill_talk",
                            "conv_ai_2",
                            "empathetic_dialogues",
                            "AlekseyKorshuk/persona-chat",
                        ], 
                        help='The dataset to use for training.')
    parser.add_argument('--output_dir',
                        type=str,
                        default='results/dialogpt',
                        help='The output directory where the model predictions and checkpoints will be written.')
    parser.add_argument('--num_train_epochs',
                        type=int,
                        default=50,
                        help='Total number of training epochs to perform.')
    parser.add_argument('--per_device_train_batch_size',
                        type=int,
                        default=10,
                        help='Batch size per GPU/CPU for training.')
    parser.add_argument('--per_device_eval_batch_size',
                        type=int,
                        default=20,
                        help='Batch size per GPU/CPU for evaluation.')
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=20,
                        help='Number of updates steps to accumulate before performing a backward/update pass.')
    parser.add_argument('--do_train',
                        action='store_true',
                        help='Whether to run training.')
    parser.add_argument('--do_eval',
                        action='store_true',
                        help='Whether to run eval on the dev set.')
    parser.add_argument('--do_predict',
                        action='store_true',
                        help='Whether to run predictions on the test set.')
    parser.add_argument('--max_train_samples',
                        type=int,
                        default=None,
                        help='For debugging purposes or quicker training, truncate the number of training examples to this.')
    parser.add_argument('--max_eval_samples',
                        type=int,
                        default=None,
                        help='For debugging purposes or quicker training, truncate the number of evaluation examples to this.')
    parser.add_argument('--max_predict_samples',
                        type=int,
                        default=None,
                        help='For debugging purposes or quicker training, truncate the number of prediction examples to this.')
    parser.add_argument('--max_length',
                        type=int,
                        default=512,
                        help='The maximum total sequence length for target text after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.')
    parser.add_argument('--pad_to_max_length',
                        action='store_true',
                        help='Whether to pad all samples to model maximum sentence length.')
    parser.add_argument('--block_size',
                        type=int,
                        default=None,
                        help='Optional input sequence length after tokenization. The training dataset will be truncated in block of this size for training.')
    parser.add_argument('--num_beams',
                        type=int,
                        default=4,
                        help='Number of beams to use for evaluation. This argument will be passed to ``model.generate``, which is used during ``evaluate`` and ``predict``.')
    parser.add_argument('--overwrite_cache',
                        action='store_true',
                        help='Overwrite the cached training and evaluation sets')
    parser.add_argument('--preprocessing_num_workers',
                        type=int,
                        default=None,
                        help='The number of processes to use for the preprocessing.')

    args = parser.parse_args()

    main(args)
