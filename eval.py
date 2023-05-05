import sys
sys.dont_write_bytecode = True
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # avoid tensorflow warnings
import re
import evaluate
from DialogueAPI import dialogue
import time
import argparse
import random
import numpy as np
from tqdm import tqdm
from typing import List, Union, Dict, Tuple
import torch
from transformers import (
    AutoConfig, 
    AutoTokenizer, 
    BartTokenizer,
    GPT2Tokenizer,
    T5Tokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
)


class EvalTransferAttack:
    def __init__(
        self,
        args: argparse.Namespace = None, 
        tokenizer: Union[BartTokenizer, GPT2Tokenizer, T5Tokenizer] = None,
        model: Union[AutoModelForSeq2SeqLM, AutoModelForCausalLM] = None,
        device: torch.device = torch.device('cpu'),
        task: str = 'seq2seq',
        bleu: evaluate.load("bleu") = None, 
        rouge: evaluate.load("rouge") = None,
        meteor: evaluate.load("meteor") = None,
    ):
        self.args = args
        self.num_beams = args.num_beams 
        self.num_beam_groups = args.num_beam_groups
        self.max_source_length = args.max_len
        self.max_target_length = args.max_len
        self.dataset = open(args.file, 'r')
        self.device = device
        
        self.task = task
        self.tokenizer = tokenizer
        self.eos_token = self.tokenizer.eos_token
        if self.task == 'seq2seq':
            self.sp_token = self.eos_token
        else:
            self.sp_token = '<SEP>'
        self.model = model.to(self.device)
        
        self.dumb_instance = {
            'history': None,
            'orig_input': None,
            'adv_input': None,
            'reference': None,
        }
        self.bleu = bleu
        self.rouge = rouge
        self.meteor = meteor
        self.ori_lens, self.adv_lens = [], []
        self.ori_bleus, self.adv_bleus = [], []
        self.ori_rouges, self.adv_rouges = [], []
        self.ori_meteors, self.adv_meteors = [], []
        self.ori_time, self.adv_time = [], []
        self.att_success = 0
        self.total_pairs = 0
        
        out_dir = args.out_dir
        dataset_n = args.dataset
        orig_model_n = args.orig_model
        victim_model_n = args.victim_model.split("/")[-1]
        
        file_path = f"{out_dir}/transfer_{orig_model_n}_{victim_model_n}_{dataset_n}.txt"
        self.write_file = open(file_path, "w")
        
        
    def log_and_save(self, display: str):
        print(display)
        self.write_file.write(display + "\n")
        
    def get_prediction(self, text: str):
        if self.task == 'seq2seq':
            effective_text = text 
        else:
            effective_text = text + self.tokenizer.eos_token

        inputs = self.tokenizer(
            effective_text,  
            return_tensors="pt",
            truncation=True,
            max_length=self.max_source_length-1,
        )
        input_ids = inputs.input_ids.to(self.device)
        t1 = time.time()
        with torch.no_grad():
            outputs = dialogue(
                self.model, 
                input_ids,
                early_stopping=False, 
                num_beams=self.num_beams,
                num_beam_groups=self.num_beam_groups, 
                use_cache=True,
                max_length=self.max_target_length,
            )
        if self.task == 'seq2seq':
            output = self.tokenizer.batch_decode(outputs['sequences'], skip_special_tokens=True)[0]
        else:
            output = self.tokenizer.batch_decode(
                outputs['sequences'][:, input_ids.shape[-1]:], 
                skip_special_tokens=True,
            )[0]
        t2 = time.time()
        return output.strip(), t2 - t1
    
    
    def eval_metrics(self, output: str, guided_messages: List[str]):
        if not output:
            return
        bleu_res = self.bleu.compute(
            predictions=[output], 
            references=[guided_messages],
            smooth=True,
        )
        rouge_res = self.rouge.compute(
            predictions=[output],
            references=[guided_messages],
        )
        meteor_res = self.meteor.compute(
            predictions=[output],
            references=[guided_messages],
        )
        pred_len = bleu_res['translation_length']
        return bleu_res, rouge_res, meteor_res, pred_len
    
        
    def eval_step(
        self,
        ins: dict,
    ):
        # Eval original
        text = ins['history'] + self.sp_token + ins['orig_input']
        references = [ins['reference']]
        self.log_and_save("\nDialogue history: {}".format(ins['history']))
        self.log_and_save("U--{} \n(Ref: ['{}', ...])".format(ins['orig_input'], references[-1]))
        output, time_gap = self.get_prediction(text)
        self.log_and_save("G--{}".format(output))
        if not output:
            return
        bleu_res, rouge_res, meteor_res, pred_len \
            = self.eval_metrics(output, references)
        self.log_and_save("(length: {}, latency: {:.3f}, BLEU: {:.3f}, ROUGE: {:.3f}, METEOR: {:.3f})".format(
            pred_len, time_gap, bleu_res['bleu'], rouge_res['rougeL'], meteor_res['meteor'],
        ))
        self.ori_lens.append(pred_len)
        self.ori_bleus.append(bleu_res['bleu'])
        self.ori_rouges.append(rouge_res['rougeL'])
        self.ori_meteors.append(meteor_res['meteor'])
        self.ori_time.append(time_gap)
            
        # Eval attack
        adv_text = ins['history'] + self.sp_token + ins['adv_input']
        adv_output, adv_time_gap = self.get_prediction(adv_text)
        self.log_and_save("U'--{}".format(ins['adv_input']))
        self.log_and_save("G'--{}".format(adv_output))
        if not adv_output:
            return
        adv_bleu_res, adv_rouge_res, adv_meteor_res, adv_pred_len \
            = self.eval_metrics(adv_output, references)
            
        # ASR
        success = (
            (bleu_res['bleu'] > adv_bleu_res['bleu']) or 
            (rouge_res['rougeL'] > adv_rouge_res['rougeL']) or 
            (meteor_res['meteor'] > adv_meteor_res['meteor'])
            )
        if success:
            self.att_success += 1
        else:
            self.log_and_save("Attack failed!")

        self.log_and_save("(length: {}, latency: {:.3f}, BLEU: {:.3f}, ROUGE: {:.3f}, METEOR: {:.3f})".format(
            adv_pred_len, adv_time_gap, adv_bleu_res['bleu'], adv_rouge_res['rougeL'], adv_meteor_res['meteor'],
        ))
        self.adv_lens.append(adv_pred_len)
        self.adv_bleus.append(adv_bleu_res['bleu'])
        self.adv_rouges.append(adv_rouge_res['rougeL'])
        self.adv_meteors.append(adv_meteor_res['meteor'])
        self.adv_time.append(adv_time_gap)
        self.total_pairs += 1
    
    
    def eval(self):
        idx = 0
        data = []
        for line in self.dataset:
            if line.startswith('Dialogue history:'):
                data.append(self.dumb_instance) # add a new instance
                history = line.split('Dialogue history:')[1].strip()
                if self.task == 'clm':
                    history = history.replace('<PS>', '').replace('<SEP>', ' ')
                # print('history:', history)
                data[idx]['history'] = history
                
            elif line.startswith('U--'):
                orig_input = line.split('U--')[1].strip()
                # print('orig input:', orig_input)
                data[idx]['orig_input'] = orig_input
                
            elif line.startswith('(Ref: ['):
                # remove the ref: and the ] at the end
                ref = line.strip()[8:-8]
                # print('reference:', ref)
                data[idx]['reference'] = ref
                
            elif line.startswith("U'--"):
                # Remove the cosine similarity in paranthesis
                adv_input = re.sub(r'\([^)]*\)', '', line.split("U'--")[1]).strip()
                # print('adv input:', adv_input)
                data[idx]['adv_input'] = adv_input
                self.eval_step(data[idx])
                idx += 1 # update the index
                
            else:
                continue
            
        Ori_len = np.mean(self.ori_lens)
        Adv_len = np.mean(self.adv_lens)
        Ori_bleu = np.mean(self.ori_bleus)
        Adv_bleu = np.mean(self.adv_bleus)
        Ori_rouge = np.mean(self.ori_rouges)
        Adv_rouge = np.mean(self.adv_rouges)
        Ori_meteor = np.mean(self.ori_meteors)
        Adv_meteor = np.mean(self.adv_meteors)
        Ori_t = np.mean(self.ori_time)
        Adv_t = np.mean(self.adv_time)

        # Summarize eval results
        self.log_and_save("\nOriginal output length: {:.3f}, latency: {:.3f}, BLEU: {:.3f}, ROUGE: {:.3f}, METEOR: {:.3f}".format(
            Ori_len, Ori_t, Ori_bleu, Ori_rouge, Ori_meteor,
        ))
        self.log_and_save("Perturbed output length: {:.3f}, latency: {:.3f}, BLEU: {:.3f}, ROUGE: {:.3f}, METEOR: {:.3f}".format(
            Adv_len, Adv_t, Adv_bleu, Adv_rouge, Adv_meteor,
        ))
        self.log_and_save("Attack success rate: {:.2f}%".format(100*self.att_success/self.total_pairs))
       
        
      
def main(args: argparse.Namespace):
    random.seed(args.seed)
    model_name_or_path = args.victim_model
    num_beams = args.num_beams
    num_beam_groups = args.num_beam_groups
    out_dir = args.out_dir

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu')
    config = AutoConfig.from_pretrained(model_name_or_path, num_beams=num_beams, num_beam_groups=num_beam_groups)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if 'gpt' in model_name_or_path.lower():
        task = 'clm'
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, config=config)
        if 'results' not in model_name_or_path.lower():
            tokenizer.add_special_tokens({'pad_token': '<PAD>'})
            tokenizer.add_special_tokens({'mask_token': '<MASK>'})
            model.resize_token_embeddings(len(tokenizer))
    else:
        task = 'seq2seq'
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, config=config)
        
    # Load evaluation metrics
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    meteor = evaluate.load("meteor")
        
    # Transfer attack
    transfer_attacker = EvalTransferAttack(
        args=args,
        tokenizer=tokenizer,
        model=model,
        device=device,
        task=task,
        bleu=bleu,
        rouge=rouge,
        meteor=meteor,
    )
    transfer_attacker.eval()

        


if __name__ == "__main__":
    import ssl
    import argparse
    # import nltk
    # nltk.download('wordnet')
    # nltk.download('omw-1.4')
    # nltk.download('averaged_perceptron_tagger')
    ssl._create_default_https_context = ssl._create_unverified_context

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", "-f", type=str, default="logging/eval_transfer_attack.txt", help="Path to file")
    parser.add_argument("--num_beams", type=int, default=2, help="Number of beams")
    parser.add_argument("--max_len", type=int, default=1024, help="Maximum length of generated sequence")
    parser.add_argument("--num_beam_groups", type=int, default=1, help="Number of beam groups")
    parser.add_argument("--orig_model", "-m", type=str, default="bart", help="Name of the original model")
    parser.add_argument("--victim_model", "-v", type=str, default="results/bart", help="Path to the victim model")
    parser.add_argument("--out_dir", type=str, default="results/logging", help="Output directory")
    parser.add_argument("--seed", type=int, default=2019, help="Random seed")
    parser.add_argument("--dataset", "-d", type=str, 
                        default="BST", 
                        choices=[
                            "BST",
                            "ConvAI2",
                            "ED",
                            "PC",
                        ], 
                        help="Dataset to attack")
    args = parser.parse_args()
    
    main(args)
