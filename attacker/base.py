import torch
import torch.nn as nn
import time
import random
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from argparse import Namespace
from DialogueAPI import dialogue
from typing import Union, List, Tuple
from transformers import ( 
    BertTokenizer,
    BertTokenizerFast,
    BartForConditionalGeneration, 
)
from utils import SentenceEncoder


class BaseAttacker:
    def __init__(
        self,     
        device: torch.device = None,
        tokenizer: Union[BertTokenizer, BertTokenizerFast] = None,
        model: BartForConditionalGeneration = None,
        max_len: int = 64,
        max_per: int = 3,
        task: str = 'seq2seq',
    ):
        self.device = device
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.embedding = self.model.get_input_embeddings().weight
        self.special_token = self.tokenizer.all_special_tokens
        self.special_id = self.tokenizer.all_special_ids
        self.eos_token = self.tokenizer.eos_token
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.num_beams = self.model.config.num_beams
        self.num_beam_groups = self.model.config.num_beam_groups
        self.max_len = max_len
        self.max_per = max_per
        self.task = task
        self.softmax = nn.Softmax(dim=1)
        self.bce_loss = nn.BCELoss()

    @classmethod
    def _get_hparam(cls, namespace: Namespace, key: str, default=None):
        if hasattr(namespace, key):
            return getattr(namespace, key)
        print('Using default argument for "{}"'.format(key))
        return default

    def run_attack(self, text: str):
        pass

    def compute_loss(self, text: list, labels: list):
        pass

    def compute_seq_len(self, seq: torch.Tensor):
        if seq.shape[0] == 0: # empty sequence
            return 0
        if seq[0].eq(self.pad_token_id):
            return int(len(seq) - sum(seq.eq(self.pad_token_id)))
        else:
            return int(len(seq) - sum(seq.eq(self.pad_token_id))) - 1

    def remove_pad(self, s: torch.Tensor):
        return s[torch.nonzero(s != self.pad_token_id)].squeeze(1)
    
    def get_prediction(self, sentence: Union[str, List[str]]):
        if self.task == 'seq2seq':
            text = sentence
        else:
            if isinstance(sentence, list):
                text = [s + self.eos_token for s in sentence]
            elif isinstance(sentence, str):
                text = sentence + self.eos_token
            else:
                raise ValueError("sentence should be a list of string or a string")

        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=self.max_len-1,
        )
        input_ids = inputs['input_ids'].to(self.device)
        # ['sequences', 'sequences_scores', 'scores', 'beam_indices']
        outputs = dialogue(
            self.model, 
            input_ids,
            early_stopping=False, 
            num_beams=self.num_beams,
            num_beam_groups=self.num_beam_groups, 
            use_cache=True,
            max_length=self.max_len,
        )
        if self.task == 'seq2seq':
            seqs = outputs['sequences'].detach()
        else:
            seqs = outputs['sequences'][:, input_ids.shape[-1]:].detach()

        seqs = [self.remove_pad(seq) for seq in seqs]
        out_scores = outputs['scores']
        pred_len = [self.compute_seq_len(seq) for seq in seqs]
        return pred_len, seqs, out_scores

    def get_cls_loss(self, sentence: List[str], labels: List[str]):
        inputs = self.tokenizer(
            sentence,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_len,
        ).to(self.device)
        labels = self.tokenizer(
            labels,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_len,
        ).to(self.device)
        if self.task == 'seq2seq':
            outputs = self.model(**inputs, labels=labels['input_ids'])
        else:
            # For clm, we need to mask the sentence (-100) and add to the labels
            new_inputs = inputs.copy()
            for k, v1 in inputs.items():
                new_inputs[k] = torch.cat((v1, labels[k]), dim=1)
            new_labels = torch.cat((-100*torch.ones_like(inputs["input_ids"]), labels["input_ids"]), dim=1)
            outputs = self.model(**new_inputs, labels=new_labels)
        
        return -outputs.loss


    def get_trans_string_len(self, text: Union[str, List[str]]):
        pred_len, seqs, _ = self.get_prediction(text)
        return seqs[0], pred_len[0]

    def get_trans_len(self, text: Union[str, List[str]]):
        pred_len, _, _ = self.get_prediction(text)
        return pred_len

    def get_trans_strings(self, text: Union[str, List[str]]):
        pred_len, seqs, _ = self.get_prediction(text)
        out_res = [self.tokenizer.decode(seq, skip_special_tokens=True) for seq in seqs]
        return out_res, pred_len

    def compute_batch_score(self, text: List[str]):
        batch_size = len(text)
        index_list = [i * self.num_beams for i in range(batch_size + 1)]
        pred_len, seqs, out_scores = self.get_prediction(text)

        scores = [[] for _ in range(batch_size)]
        for out_s in out_scores:
            for i in range(batch_size):
                current_index = index_list[i]
                scores[i].append(out_s[current_index: current_index + 1])
        scores = [torch.cat(s) for s in scores]
        scores = [s[:pred_len[i]] for i, s in enumerate(scores)]
        return scores, seqs, pred_len
    
    def compute_score(self, text: Union[str, List[str]], batch_size: int = None):
        total_size = len(text)
        if batch_size is None:
            batch_size = len(text)

        if batch_size < total_size:
            scores, seqs, pred_len = [], [], []
            for start in range(0, total_size, batch_size):
                end = min(start + batch_size, total_size)
                score, seq, p_len = self.compute_batch_score(text[start: end])
                pred_len.extend(p_len)
                seqs.extend(seq)
                scores.extend(score)
        else:
            scores, seqs, pred_len = self.compute_batch_score(text)
        return scores, seqs, pred_len



class SlowAttacker(BaseAttacker):
    def __init__(
        self, 
        device: torch.device = None,
        tokenizer: Union[BertTokenizer, BertTokenizerFast] = None,
        model: BartForConditionalGeneration = None,
        max_len: int = 64,
        max_per: int = 3,
        task: str = 'seq2seq',
        fitness: str = 'performance',
        select_beams: int = 1,
        eos_weight: float = 0.5,
        cls_weight: float = 0.5,
        use_combined_loss: bool = False,
    ):
        super(SlowAttacker, self).__init__(
            device, tokenizer, model, max_len, max_per, task,
        )
        if self.task == 'seq2seq':
            self.sp_token = self.eos_token
        else:
            self.sp_token = '<SEP>'
        self.select_beam = select_beams
        self.fitness = fitness
        self.eos_weight = eos_weight
        self.cls_weight = cls_weight
        self.use_combined_loss = use_combined_loss
        self.sent_encoder = SentenceEncoder(device='cpu')

    def leave_eos_loss(self, scores: list, pred_len: list):
        loss = []
        for i, s in enumerate(scores):
            s[:, self.pad_token_id] = 1e-12 # T X V
            eos_p = self.softmax(s)[:pred_len[i], self.eos_token_id]
            loss.append(self.bce_loss(eos_p, torch.zeros_like(eos_p)))
        return loss

    def leave_eos_target_loss(self, scores: list, seqs: list, pred_len: list):
        loss = []
        for i, s in enumerate(scores): # s: T X V
            if pred_len[i] == 0:
                loss.append(torch.tensor(0.0, requires_grad=True).to(self.device))
            else:
                s[:, self.pad_token_id] = 1e-12
                softmax_v = self.softmax(s)
                eos_p = softmax_v[:pred_len[i], self.eos_token_id]
                target_p = torch.stack([softmax_v[idx, v] for idx, v in enumerate(seqs[i][1:])])
                target_p = target_p[:pred_len[i]]
                pred = eos_p + target_p
                pred[-1] = pred[-1] / 2
                loss.append(self.bce_loss(pred, torch.zeros_like(pred)))
        return loss

    def get_target_p(self, scores: list, pred_len: list, label: list):
        targets = []
        for i, s in enumerate(scores): # s: T X V
            if pred_len[i] == 0:
                targets.append(torch.tensor(0.0).to(self.device))
            else:
                # if self.pad_token_id != self.eos_token_id:
                s[:, self.pad_token_id] = 1e-12
                softmax_v = self.softmax(s) # T X V
                target_p = torch.stack([softmax_v[idx, v] for idx, v in enumerate(label[:softmax_v.size(0)])])
                target_p = target_p[:pred_len[i]]
                targets.append(torch.sum(target_p))
        return torch.stack(targets).detach().cpu().numpy()

    @torch.no_grad()
    def select_best(
        self, 
        new_strings: List[Tuple], 
        goal: str,
        batch_size: int = 3,
        it: int = 0, 
        prototype: str = None,
    ):
        """
        Select generated strings which induce longest output sentences.
        """
        pred_len, pred_acc = [], []
        batch_num = len(new_strings) // batch_size
        if batch_size * batch_num != len(new_strings):
            batch_num += 1
            
        for i in range(batch_num):
            st, ed = i * batch_size, min(i * batch_size + batch_size, len(new_strings))
            if self.task == 'seq2seq':
                batch_strings = [x[1] for x in new_strings[st:ed]]
            else:
                batch_strings = [x[1] + self.eos_token for x in new_strings[st:ed]]
            if not batch_strings:
                continue
            scores, seqs, p_len = self.compute_batch_score(batch_strings)
            pred_len.extend(p_len)
            if self.fitness in ['performance', 'combined', 'adaptive']:
                label = self.tokenizer(goal, truncation=True, max_length=self.max_len, return_tensors='pt')
                label = label['input_ids'][0] # (T, )
                res = self.get_target_p(scores, p_len, label) # numpy array (N, )
                pred_acc.extend(res.tolist())
            
        pred_len = torch.tensor(pred_len) # (#new strings, )
        pred_acc = torch.tensor(pred_acc) # (#new strings, )
        assert len(new_strings) == len(pred_len)
        if self.fitness == 'length':
            top_v, top_i = pred_len.topk(min(self.select_beam, len(pred_len)))
            return [new_strings[i] for i in top_i], top_v
        elif self.fitness == 'performance':
            assert len(new_strings) == len(pred_acc)
            top_v, top_i = pred_acc.topk(min(self.select_beam, len(pred_acc)), largest=False)
            return [new_strings[i] for i in top_i], [pred_len[i] for i in top_i]
        elif self.fitness == 'combined': 
            assert len(pred_len) == len(pred_acc)
            pred_comb = pred_len / (pred_acc + 1e-7)
            top_v, top_i = pred_comb.topk(min(self.select_beam, len(pred_acc)))
            return [new_strings[i] for i in top_i], [pred_len[i] for i in top_i]
        elif self.fitness == 'random': # random
            rand_i = np.random.choice(len(pred_len), min(self.select_beam, len(pred_len)), replace=False)
            return [new_strings[i] for i in rand_i], [pred_len[i] for i in rand_i]
        elif self.fitness == 'adaptive':
            assert len(pred_len) == len(pred_acc)
            q = np.mean([self.sent_encoder.get_sim(prototype, x[1]) for x in new_strings])
            preference = (it - 1) * np.exp(q - 1) / (self.max_per - 1)
            if preference > 0.5:
                # Random search
                rand_i = np.random.choice(len(pred_len), min(self.select_beam, len(pred_len)), replace=False)
                return [new_strings[i] for i in rand_i], [pred_len[i] for i in rand_i]
            else:
                # Greedy search
                pred_comb = pred_len / (pred_acc + 1e-7)
                top_v, top_i = pred_comb.topk(min(self.select_beam, len(pred_acc)))
                return [new_strings[i] for i in top_i], [pred_len[i] for i in top_i]
        else:
            raise NotImplementedError
            
        

    def prepare_attack(self, text: Union[str, List[str]]):
        ori_len = self.get_trans_len(text)[0] # original sentence length
        best_adv_text, best_len = deepcopy(text), ori_len
        cur_adv_text, cur_len = deepcopy(text), ori_len  # current_adv_text: List[str]
        return ori_len, (best_adv_text, best_len), (cur_adv_text, cur_len)

    def compute_loss(self, text: list, labels: list):
        raise NotImplementedError

    def mutation(self, cur_adv_text: str, grad: torch.gradient, modified_pos: List[int]):
        raise NotImplementedError

    def pareto_step(self, weights_list: np.ndarray, out_gradients_list: np.ndarray):
        model_gradients = out_gradients_list
        M1 = np.matmul(model_gradients,np.transpose(model_gradients)) # 2 x 2
        e = np.mat(np.ones(np.shape(weights_list)))
        M = np.hstack((M1,np.transpose(e)))
        mid = np.hstack((e,np.mat(np.zeros((1,1)))))
        M = np.vstack((M,mid)) # 3 x 3
        z = np.mat(np.zeros(np.shape(weights_list)))
        nid = np.hstack((z,np.mat(np.ones((1,1))))) # 1 x 3
        w = np.matmul(np.matmul(M,np.linalg.inv(np.matmul(M,np.transpose(M)))),np.transpose(nid))
        if len(w) > 1:
            w = np.transpose(w)
            w = w[0,0:np.shape(w)[1]]
            mid = np.where(w > 0, 1.0, 0)
            nid = np.multiply(mid, w)
            uid = sorted(nid[0].tolist()[0], reverse=True)
            sv = np.cumsum(uid)
            rho = np.where(uid > (sv - 1.0) / range(1, len(uid)+1), 1.0, 0.0)
            r = max(np.argwhere(rho))
            theta = max(0, (sv[r] - 1.0) / (r+1))
            w = np.where(nid - theta>0.0, nid - theta, 0)
        return w
    
    def calibrate_grads(self, grad1: torch.gradient, grad2: torch.gradient, w1: float, w2: float):
        weights = np.mat([w1, w2])
        grad_paras_tensor = torch.stack((grad1.view(-1), grad2.view(-1)), dim=0) # (2, V * H)
        grad_paras = grad_paras_tensor.detach().cpu().numpy()
        mid = self.pareto_step(weights, grad_paras)
        new_w1, new_w2 = mid[0,0], mid[0,1]
        return new_w1, new_w2
    

    def run_attack(self, text: str, label: str):
        """
        Inputs:
            text: original sentence (context + sp_token + free message)
            label: guided message
        Outputs:
            success: whether the attack is successful
            adv_his: list of tuples (adv sentence, adv length, decoding time)
            
        (1) Using gradient ascent to generate adversarial sentences -- mutation();
        (2) Select the best samples which induce longest output sentences -- select_best();
        (3) Save the adversarial samples -- adv_his.
        """
        torch.autograd.set_detect_anomaly(True)
        ori_len, (best_adv_text, best_len), (cur_adv_text, cur_len) = self.prepare_attack(text)
        ori_context = cur_adv_text.split(self.sp_token)[0].strip()
        adv_his = []
        modify_pos = [] # record already modified positions (avoid repeated perturbation)
        t1 = time.time()
            
        def generate_new_strings(cur_text: str, w1: float, w2: float):
            """
            Args:
                cur_text (str): context + sp_token + current adversarial free message
                w1 (float): weight for cls_loss
                w2 (float): weight for eos_loss
            Returns:
                new_strings (list): list of (pos, new adversarial sentence)
                new_w1 (float): calibrated weight for cls_loss
                new_w2 (float): calibrated weight for eos_loss
            """
            loss_list, cls_loss = self.compute_loss([cur_text], [label])
            if cls_loss is not None:
                self.model.zero_grad()
                cls_loss.backward(retain_graph=True)
                grad1 = self.embedding.grad.clone()
            else:
                grad1 = None
                
            if loss_list is not None:
                eos_loss = sum(loss_list)
                self.model.zero_grad()
                eos_loss.backward(retain_graph=True)
                grad2 = self.embedding.grad.clone()
            else:
                grad2 = None
            
            if (grad1 is not None) and (grad2 is not None):
                if self.use_combined_loss:
                    new_w1, new_w2 = self.calibrate_grads(grad1, grad2, w1, w2)
                    loss = new_w1 * cls_loss + new_w2 * eos_loss
                    self.model.zero_grad()
                    loss.backward()
                    grad = self.embedding.grad
                else:
                    grad = grad2
                    new_w1, new_w2 = w1, w2
            elif grad1 is not None:
                grad = grad1
                new_w1, new_w2 = w1, w2
            elif grad2 is not None:
                grad = grad2
                new_w1, new_w2 = w1, w2
            else:
                grad = None
                new_w1, new_w2 = w1, w2
            
            # Only mutate the part after special token
            cur_free_text = cur_text.split(self.sp_token)[1].strip()
            new_strings = self.mutation(ori_context, cur_free_text, grad, label, modify_pos)
            # Pad the original context
            new_strings = [
                (pos, ori_context + self.sp_token + adv_text)
                for (pos, adv_text) in new_strings
            ]
            return new_strings, new_w1, new_w2


        def get_best_adv(
            it: int, 
            cur_sent: str, 
            w1: float, 
            w2: float, 
            start: float, 
            best_text: str, 
            best_length: int, 
            modified_pos: List[int], 
            adv_history: List[tuple],
        ):
            if it == self.max_per:
                return best_text, best_length, adv_history
            # Get new strings
            new_strings, new_w1, new_w2 = generate_new_strings(cur_sent, w1, w2)
            # Select the best strings
            cur_topk_strings, cur_lens = self.select_best(new_strings, label, batch_size=3, it=it, prototype=ori_context)
            # print("\n[it {}] \ncur sent: {} \nnew_string: {} \ncur_strings: {}".format(it, cur_sent, new_strings, cur_topk_strings))
            end = time.time()
            # Beam search
            for i in range(len(cur_topk_strings)):
                cur_pos, cur_text = cur_topk_strings[i]
                cur_len = cur_lens[i]
                if isinstance(cur_pos, list):
                    modified_pos.extend(cur_pos)
                else:
                    modified_pos.append(int(cur_pos))
                if cur_len > best_length:
                    best_text = str(cur_text)
                    best_length = int(cur_len)
                print("[iteration %d][sent %d][len %d (%.2f)] %s" % (
                    it, i, cur_len, cur_len/ori_len, cur_text.split(self.sp_token)[1].strip()
                ))
                adv_history.append((deepcopy(best_text), int(best_length), end - start))
                best_text, best_length, adv_history = get_best_adv(
                    it+1, cur_text, new_w1, new_w2, end, best_text, best_length, modified_pos, adv_history
                )
            return best_text, best_length, adv_history

        get_best_adv(0, cur_adv_text, self.cls_weight, self.eos_weight, t1, best_adv_text, best_len, modify_pos, adv_his)
        if adv_his:
            return True, adv_his
        else:
            return False, [(deepcopy(cur_adv_text), deepcopy(cur_len), 0.0)]


