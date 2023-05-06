import sys
sys.dont_write_bytecode = True
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # avoid tensorflow warnings
import time
import argparse
import random
import numpy as np
from tqdm import tqdm
from typing import List
import torch
from transformers import (
    AutoConfig, 
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
)
from datasets import load_dataset, Dataset
import evaluate
from DialogueAPI import dialogue
from attacker.DGSlow import WordAttacker, StructureAttacker
from attacker.PWWS import PWWSAttacker
from attacker.SCPN import SCPNAttacker
from attacker.VIPER import VIPERAttacker
from attacker.BAE import BAEAttacker
from attacker.FD import FDAttacker
from attacker.HotFlip import HotFlipAttacker
from attacker.TextBugger import TextBuggerAttacker
from attacker.MAYA import MAYAAttacker
from attacker.UAT import UATAttacker
from DG_dataset import DGDataset

DATA2NAME = {
    "blended_skill_talk": "BST",
    "conv_ai_2": "ConvAI2",
    "empathetic_dialogues": "ED",
    "AlekseyKorshuk/persona-chat": "PC",
}

class DGAttackEval(DGDataset):
    def __init__(
        self, 
        args: argparse.Namespace = None, 
        tokenizer: AutoTokenizer = None, 
        model: AutoModelForSeq2SeqLM = None, 
        attacker: WordAttacker = None, 
        device: torch.device('cpu') = None, 
        task: str = 'seq2seq', 
        bleu: evaluate.load("bleu") = None, 
        rouge: evaluate.load("rouge") = None,
        meteor: evaluate.load("meteor") = None,
    ):
        super(DGAttackEval, self).__init__(
            dataset=args.dataset,
            task=task,
            tokenizer=tokenizer,
            max_source_length=args.max_len,
            max_target_length=args.max_len,
            padding=None,
            ignore_pad_token_for_loss=True,
            preprocessing_num_workers=None,
            overwrite_cache=True,
        )

        self.args = args
        self.model = model
        self.attacker = attacker
        self.device = device
        self.sp_token = attacker.sp_token

        self.num_beams = args.num_beams 
        self.num_beam_groups = args.num_beam_groups
        self.max_num_samples = args.max_num_samples

        self.bleu = bleu
        self.rouge = rouge
        self.meteor = meteor

        self.ori_lens, self.adv_lens = [], []
        self.ori_bleus, self.adv_bleus = [], []
        self.ori_rouges, self.adv_rouges = [], []
        self.ori_meteors, self.adv_meteors = [], []
        self.ori_time, self.adv_time = [], []
        self.cos_sims = []
        self.att_success = 0
        self.total_pairs = 0
        
        # self.record = []
        att_method = args.attack_strategy
        out_dir = args.out_dir
        model_n = args.model_name_or_path.split("/")[-1]
        dataset_n = DATA2NAME.get(args.dataset, args.dataset.split("/")[-1])
        combined = "combined" if args.use_combined_loss and att_method == 'structure' else "single"
        max_per = args.max_per
        fitness = args.fitness if att_method == 'structure' else 'performance'
        select_beams = args.select_beams if att_method == 'structure' else 1
        max_num_samples = args.max_num_samples
        file_path = f"{out_dir}/{combined}_{att_method}_{max_per}_{fitness}_{select_beams}_{model_n}_{dataset_n}_{max_num_samples}.txt"
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
        
        
    def generation_step(self, instance: dict):
        # Set up
        num_entries, total_entries, context, prev_utt_pc = self.prepare_context(instance)
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
            
            prev_utt_pc += [
                free_message,
                guided_message,
            ]
            self.log_and_save("\nDialogue history: {}".format(original_context))
            self.log_and_save("U--{} \n(Ref: ['{}', ...])".format(free_message, references[-1]))
            # Original generation
            text = original_context + self.sp_token + free_message
            output, time_gap = self.get_prediction(text)
            self.log_and_save("G--{}".format(output))
            
            if not output:
                continue
            bleu_res, rouge_res, meteor_res, pred_len = self.eval_metrics(output, references)
            self.log_and_save("(length: {}, latency: {:.3f}, BLEU: {:.3f}, ROUGE: {:.3f}, METEOR: {:.3f})".format(
                pred_len, time_gap, bleu_res['bleu'], rouge_res['rougeL'], meteor_res['meteor'],
            ))
            self.ori_lens.append(pred_len)
            self.ori_bleus.append(bleu_res['bleu'])
            self.ori_rouges.append(rouge_res['rougeL'])
            self.ori_meteors.append(meteor_res['meteor'])
            self.ori_time.append(time_gap)
            
            # Attack
            success, adv_his = self.attacker.run_attack(text, guided_message)
            new_text = adv_his[-1][0]
            new_free_message = new_text.split(self.sp_token)[1].strip()
            cos_sim = self.attacker.sent_encoder.get_sim(new_free_message, free_message)
            output, time_gap = self.get_prediction(new_text)
            if not output:
                continue

            self.log_and_save("U'--{} (cosine: {:.3f})".format(new_free_message, cos_sim))
            self.log_and_save("G'--{}".format(output))
            adv_bleu_res, adv_rouge_res, adv_meteor_res, adv_pred_len = self.eval_metrics(output, references)
            
            # ASR
            success = (
                (bleu_res['bleu'] > adv_bleu_res['bleu']) or 
                (rouge_res['rougeL'] > adv_rouge_res['rougeL']) or 
                (meteor_res['meteor'] > adv_meteor_res['meteor'])
                ) and cos_sim > 0.01
            if success:
                self.att_success += 1
            else:
                self.log_and_save("Attack failed!")

            self.log_and_save("(length: {}, latency: {:.3f}, BLEU: {:.3f}, ROUGE: {:.3f}, METEOR: {:.3f})".format(
                adv_pred_len, time_gap, adv_bleu_res['bleu'], adv_rouge_res['rougeL'], adv_meteor_res['meteor'],
            ))
            self.adv_lens.append(adv_pred_len)
            self.adv_bleus.append(adv_bleu_res['bleu'])
            self.adv_rouges.append(adv_rouge_res['rougeL'])
            self.adv_meteors.append(adv_meteor_res['meteor'])
            self.adv_time.append(time_gap)
            self.cos_sims.append(cos_sim)
            self.total_pairs += 1


    def generation(self, test_dataset: Dataset):
        if self.dataset == "empathetic_dialogues":
            test_dataset = self.group_ED(test_dataset)

        # Sample test dataset
        ids = random.sample(range(len(test_dataset)), self.max_num_samples)
        test_dataset = test_dataset.select(ids)
        print("Test dataset: ", test_dataset)
        for i, instance in tqdm(enumerate(test_dataset)):
            self.generation_step(instance)

        Ori_len = np.mean(self.ori_lens)
        Adv_len = np.mean(self.adv_lens)
        Ori_bleu = np.mean(self.ori_bleus)
        Adv_bleu = np.mean(self.adv_bleus)
        Ori_rouge = np.mean(self.ori_rouges)
        Adv_rouge = np.mean(self.adv_rouges)
        Ori_meteor = np.mean(self.ori_meteors)
        Adv_meteor = np.mean(self.adv_meteors)
        Cos_sims = np.mean(self.cos_sims)
        Ori_t = np.mean(self.ori_time)
        Adv_t = np.mean(self.adv_time)

        # Summarize eval results
        self.log_and_save("\nOriginal output length: {:.3f}, latency: {:.3f}, BLEU: {:.3f}, ROUGE: {:.3f}, METEOR: {:.3f}".format(
            Ori_len, Ori_t, Ori_bleu, Ori_rouge, Ori_meteor,
        ))
        self.log_and_save("Perturbed [cosine: {:.3f}] output length: {:.3f}, latency: {:.3f}, BLEU: {:.3f}, ROUGE: {:.3f}, METEOR: {:.3f}".format(
            Cos_sims, Adv_len, Adv_t, Adv_bleu, Adv_rouge, Adv_meteor,
        ))
        self.log_and_save("Attack success rate: {:.2f}%".format(100*self.att_success/self.total_pairs))



def main(args: argparse.Namespace):
    random.seed(args.seed)
    model_name_or_path = args.model_name_or_path
    dataset = args.dataset
    max_len = args.max_len
    max_per = args.max_per
    num_beams = args.num_beams
    select_beams = args.select_beams
    fitness = args.fitness
    num_beam_groups = args.num_beam_groups
    att_method = args.attack_strategy
    cls_weight = args.cls_weight
    eos_weight = args.eos_weight
    delta = args.delta
    use_combined_loss = args.use_combined_loss
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

    # Load dataset
    all_datasets = load_dataset(dataset)
    if dataset == "conv_ai_2":
        test_dataset = all_datasets['train']
    elif dataset == "AlekseyKorshuk/persona-chat":
        test_dataset = all_datasets['validation']
    else:
        test_dataset = all_datasets['test']

    # Define attack method
    if att_method.lower() == 'word':
        attacker = WordAttacker(
            device=device,
            tokenizer=tokenizer,
            model=model,
            max_len=max_len,
            max_per=max_per,
            task=task,
        )
    elif att_method.lower() == 'structure':
        attacker = StructureAttacker(
            device=device,
            tokenizer=tokenizer,
            model=model,
            max_len=max_len,
            max_per=max_per,
            task=task,
            fitness=fitness,
            select_beams=select_beams,
            eos_weight=eos_weight,
            cls_weight=cls_weight,
            delta=delta,
            use_combined_loss=use_combined_loss,
        )
    elif att_method.lower() == 'pwws':
        attacker = PWWSAttacker(
            device=device,
            tokenizer=tokenizer,
            model=model,
            max_len=max_len,
            max_per=max_per,
            task=task,
        )
    elif att_method.lower() == 'scpn':
        attacker = SCPNAttacker(
            device=device,
            tokenizer=tokenizer,
            model=model,
            max_len=max_len,
            max_per=max_per,
            task=task,
        )
    elif att_method.lower() == 'viper':
        attacker = VIPERAttacker(
            device=device,
            tokenizer=tokenizer,
            model=model,
            max_len=max_len,
            max_per=max_per,
            task=task,
        )
    elif att_method.lower() == 'bae':
        attacker = BAEAttacker(
            device=device,
            tokenizer=tokenizer,
            model=model,
            max_len=max_len,
            max_per=max_per,
            task=task,
        )
    elif att_method.lower() == 'fd':
        attacker = FDAttacker(
            device=device,
            tokenizer=tokenizer,
            model=model,
            max_len=max_len,
            max_per=max_per,
            task=task,
        )
    elif att_method.lower() == 'hotflip':
        attacker = HotFlipAttacker(
            device=device,
            tokenizer=tokenizer,
            model=model,
            max_len=max_len,
            max_per=max_per,
            task=task,
        )
    elif att_method.lower() == 'textbugger':
        attacker = TextBuggerAttacker(
            device=device,
            tokenizer=tokenizer,
            model=model,
            max_len=max_len,
            max_per=max_per,
            task=task,
        ) 
    elif att_method.lower() == 'maya':
        attacker = MAYAAttacker(
            device=device,
            tokenizer=tokenizer,
            model=model,
            max_len=max_len,
            max_per=max_per,
            task=task,
        ) 
    elif att_method.lower() == 'uat':
        attacker = UATAttacker(
            device=device,
            tokenizer=tokenizer,
            model=model,
            max_len=max_len,
            max_per=max_per,
            task=task,
        ) 
    else:
        raise ValueError("Invalid attack strategy!")

    # Load evaluation metrics
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    meteor = evaluate.load("meteor")

    # Define DG attack
    dg = DGAttackEval(
        args=args,
        tokenizer=tokenizer,
        model=model,
        attacker=attacker,
        device=device,
        task=task,
        bleu=bleu,
        rouge=rouge,
        meteor=meteor,
    )
    dg.generation(test_dataset)

    # # Save generation files
    # model_n = model_name_or_path.split("/")[-1]
    # dataset_n = DATA2NAME.get(dataset, dataset.split("/")[-1])
    # combined = "combined" if use_combined_loss else "eos"
    # file_path = f"{out_dir}/{combined}_{att_method}_{max_per}_{fitness}_{select_beams}_{model_n}_{dataset_n}_{max_num_samples}.txt"
    # with open(file_path, "w") as f:
    #     for line in dg.record:
    #         f.write(str(line) + "\n")
    # f.close()


if __name__ == "__main__":
    import ssl
    import argparse
    # import nltk
    # nltk.download('wordnet')
    # nltk.download('omw-1.4')
    # nltk.download('averaged_perceptron_tagger')
    ssl._create_default_https_context = ssl._create_unverified_context

    parser = argparse.ArgumentParser()
    parser.add_argument("--max_num_samples", type=int, default=5, help="Number of samples to attack")
    parser.add_argument("--max_per", type=int, default=5, help="Number of perturbation iterations per sample")
    parser.add_argument("--max_len", type=int, default=1024, help="Maximum length of generated sequence")
    parser.add_argument("--select_beams", type=int, default=2, help="Number of sentence beams to keep for each attack iteration")
    parser.add_argument("--num_beams", type=int, default=1, help="Number of beams for decoding in LLMs")
    parser.add_argument("--num_beam_groups", type=int, default=1, help="Number of beam groups for decoding in LLMs")
    parser.add_argument("--fitness", type=str, default="adaptive", 
                        choices=["performance", "length", "random", "combined", "adaptive"], 
                        help="Fitness function for selecting the best candidate")
    parser.add_argument("--model_name_or_path", "-m", type=str, default="results/bart", help="Path to model")
    parser.add_argument("--dataset", "-d", type=str, default="blended_skill_talk", 
                        choices=["blended_skill_talk", "conv_ai_2", "empathetic_dialogues", "AlekseyKorshuk/persona-chat"], 
                        help="Dataset to attack")
    parser.add_argument("--out_dir", type=str,
                        default="results/logging",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=2019, help="Random seed")
    parser.add_argument("--eos_weight", type=float, default=0.8, help="Weight for EOS gradient")
    parser.add_argument("--cls_weight", type=float, default=0.2, help="Weight for classification gradient")
    parser.add_argument("--delta", type=float, default=0.5, help="Threshold for adaptive search strategy")
    parser.add_argument("--use_combined_loss", action="store_true", help="Use combined loss")
    parser.add_argument("--attack_strategy", "-a", type=str, 
                        default='structure', 
                        choices=[
                            'structure', 
                            'word', 
                            # 'pwws', 
                            # 'scpn', 
                            # 'viper',
                            # 'bae',
                            'fd', # white-box attack
                            'hotflip', # white-box attack 
                            'textbugger', # white-box attack
                            'uat', # white-box attack
                            # 'maya',
                            ], 
                        help="Attack strategy")
    args = parser.parse_args()
    main(args)
