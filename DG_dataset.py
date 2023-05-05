import torch
from datasets import load_dataset, Dataset
from itertools import chain
from typing import List, Optional
from transformers import AutoTokenizer
from transformers.testing_utils import CaptureLogger
from transformers.utils.logging import get_logger


class DGDataset:
    def __init__(
        self, 
        dataset: str = "blended_skill_talk",
        task: str = "seq2seq",
        tokenizer: AutoTokenizer = None,
        max_source_length: int = 512,
        max_target_length: int = 512,
        padding: str = "max_length",
        ignore_pad_token_for_loss: bool = True,
        preprocessing_num_workers: int = None,
        overwrite_cache: bool = True,
    ):
        self.dataset = dataset
        self.task = task
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.padding = padding
        self.ignore_pad_token_for_loss = ignore_pad_token_for_loss
        self.preprocessing_num_workers = preprocessing_num_workers
        self.overwrite_cache = overwrite_cache
        # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
        self.tok_logger = get_logger("transformers.tokenization_utils_base")


    def prepare_context(self, instance: dict):
        if self.dataset == 'blended_skill_talk':
            num_entries = len(instance["free_messages"])
            total_entries = num_entries
            if self.task == 'seq2seq':
                persona_pieces = f"<PS>{instance['personas'][1]}"
                if instance['context'] == "wizard_of_wikipedia":
                    additional_context_pieces = f"<CTX>{instance['additional_context']}."
                else:
                    additional_context_pieces = ""
                context = persona_pieces + additional_context_pieces
            else:
                num_entries = min(num_entries, 2)
                context = ''
            prev_utt_pc = [sent for sent in instance["previous_utterance"] if sent != '']

        elif self.dataset == 'conv_ai_2':
            total_entries = len(instance['dialog'])
            num_entries = total_entries//2
            if self.task == 'seq2seq':
                user_profile = ' '.join([''.join(x) for x in instance['user_profile']])
                persona_pieces = f"<PS>{user_profile}"
                context = persona_pieces
            else:
                num_entries = min(num_entries, 2)
                context = ''
            prev_utt_pc = []

        elif self.dataset == 'empathetic_dialogues':
            total_entries = len(instance['dialog'])
            num_entries = total_entries//2
            if self.task == 'seq2seq':
                persona_pieces = f"<PS>{instance['prompt']}"
                additional_context_pieces = f"<CTX>{instance['context']}."
                context = persona_pieces + additional_context_pieces
            else:
                num_entries = min(num_entries, 2)
                context = ''
            prev_utt_pc = []

        elif self.dataset == 'AlekseyKorshuk/persona-chat':
            total_entries = len(instance['utterances'])
            num_entries = total_entries//2
            if self.task == 'seq2seq':
                user_profile = ' '.join(instance['personality'])
                persona_pieces = f"<PS>{user_profile}"
                context = persona_pieces
            else:
                num_entries = min(num_entries, 2)
                context = ''
            prev_utt_pc = []

        else:
            raise ValueError("Dataset not supported.")
        return num_entries, total_entries, context, prev_utt_pc


    def prepare_entry(
        self, 
        instance: dict, 
        entry_idx: int, 
        context: str, 
        prev_utt_pc: List[str], 
        total_entries: int,
    ):
        if self.dataset == 'blended_skill_talk':
            free_message = instance['free_messages'][entry_idx]
            guided_message = instance['guided_messages'][entry_idx]
            references = [values[entry_idx] for key, values in instance['suggestions'].items()]

        elif self.dataset == 'conv_ai_2':
            free_message = instance['dialog'][entry_idx*2]['text']
            if entry_idx*2+1 >= total_entries:
                guided_message = None
            else:
                guided_message = instance['dialog'][entry_idx*2+1]['text']
            references = []

        elif self.dataset == 'empathetic_dialogues':
            free_message = instance['dialog'][entry_idx*2]['text']
            if entry_idx*2+1 >= total_entries:
                guided_message = None
            else:
                guided_message = instance['dialog'][entry_idx*2+1]['text']
            references = []

        elif self.dataset == 'AlekseyKorshuk/persona-chat':
            free_message = instance['utterances'][entry_idx*2]['history'][-1]
            if entry_idx*2+1 >= total_entries:
                guided_message = None
            else:
                guided_message = instance['utterances'][entry_idx*2+1]['history'][-1]
            references = instance['utterances'][entry_idx*2]['candidates']
            
        else:
            raise ValueError("Dataset not supported.")

        if not prev_utt_pc:
            original_context = context
        else:
            sp_token = '<SEP>' if self.task == 'seq2seq' else ' '
            original_context = context + sp_token + sp_token.join(prev_utt_pc)
        
        references.append(guided_message)
        return free_message, guided_message, original_context, references


    def tokenize_and_align_labels(self, instance: dict):
        num_entries, total_entries, context, prev_utt_pc = self.prepare_context(instance)
        inputs, labels = [], []
        for entry_idx in range(num_entries):
            free_message, guided_message, original_context, references = self.prepare_entry(
                instance, 
                entry_idx, 
                context, 
                prev_utt_pc,
                total_entries,
            )
            if guided_message is None:
                continue
            # Input & Output
            if self.task == 'seq2seq':
                text = original_context + self.tokenizer.eos_token + free_message
            else:
                text = original_context + free_message + guided_message

            inputs.append(text)
            labels.append(guided_message)
            prev_utt_pc += [
                free_message,
                guided_message,
            ]
        
        if not inputs:
            return {"input_ids": [], "labels": [], "attention_mask": []}

        if self.task == 'seq2seq':
            inputs = self.tokenizer(inputs, max_length=self.max_source_length, padding=self.padding, truncation=True)
            # Setup the tokenizer for targets
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(labels, max_length=self.max_target_length, padding=self.padding, truncation=True)
            
            # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 
            # when we want to ignore padding in the loss.
            if self.padding == "max_length" and self.ignore_pad_token_for_loss:
                labels["input_ids"] = [
                    [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
                ]
            inputs["labels"] = labels["input_ids"]
            return inputs
        else:
            with CaptureLogger(self.tok_logger) as cl:
                inputs = self.tokenizer(
                    inputs, 
                    return_tensors="pt",
                    max_length=self.max_source_length, 
                    padding=self.padding, 
                    truncation=True,
                )
                labels = self.tokenizer(
                    labels, 
                    return_tensors="pt",
                    max_length=self.max_target_length, 
                    padding=self.padding, 
                    truncation=True,
                )
                
            new_inputs = inputs.copy()
            for k, v1 in inputs.items():
                v2 = labels[k]
                new_inputs[k] = torch.cat((v1, v2), dim=1)
                
            new_labels = torch.cat((-100*torch.ones_like(inputs["input_ids"]), labels["input_ids"]), dim=1)
            new_inputs["labels"] = new_labels

            # clm input could be much much longer than block_size
            if "Token indices sequence length is longer than the" in cl.out:
                self.tok_logger.warning(
                    "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                    " before being passed to the model."
                )
            return new_inputs


    def group_texts(self, examples):
        # ['input_ids', 'attention_mask', 'labels']
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        return concatenated_examples


    def group_ED(self, dataset: Dataset):
        results = {
            'conv_id': [], 
            'prompt': [],
            'dialog': [], 
            'context': [],
        }
        for i, instance in enumerate(dataset):
            if instance['utterance_idx'] == 1:
                results['conv_id'].append(instance['conv_id'])
                results['dialog'].append([])
                results['prompt'].append(instance['prompt'])
                results['context'].append(instance['context'])

            response = {'text': instance['utterance'], 'speaker_idx': instance['speaker_idx']}
            results['dialog'][-1].append(response)
        return Dataset.from_dict(results)


    def preprocess(self, dataset: Dataset):
        if self.dataset == "empathetic_dialogues":
            dataset = self.group_ED(dataset)

        dataset = dataset.map(
            self.tokenize_and_align_labels,
            batched=False,
            num_proc=self.preprocessing_num_workers,
            remove_columns=dataset.column_names,
            load_from_cache_file=not self.overwrite_cache,
        )
        dataset = dataset.map(
            self.group_texts,
            batched=True,
            num_proc=self.preprocessing_num_workers,
            load_from_cache_file=not self.overwrite_cache,
        )
        return dataset

    


if __name__ == "__main__":
    from transformers import AutoTokenizer
    from datasets import load_dataset
    
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    tokenizer.pad_token = tokenizer.eos_token
    data_names = [
        "conv_ai_2",
        "empathetic_dialogues",
        "AlekseyKorshuk/persona-chat",
        "blended_skill_talk",
    ]
    task = "seq2seq"
    max_length = 256
    
    for data_name in data_names:
        train_dataset = load_dataset(data_name)["train"]
        dg = DGDataset(
            dataset=data_name,
            task=task,
            tokenizer=tokenizer,
            max_source_length=max_length,
            max_target_length=max_length,
        )
        print('{}: {}'.format(data_name, train_dataset))
        train_dataset = dg.preprocess(train_dataset)
        print("processed dataset: ", train_dataset)
        print("processed dataset[0]: ", train_dataset[0])
        


        