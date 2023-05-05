import torch
import numpy as np
from typing import Optional, Union, List
from utils import ENGLISH_FILTER_WORDS
from .base import SlowAttacker
from transformers import (
    BertTokenizerFast, 
    BertForMaskedLM,
    BartForConditionalGeneration,
)
from OpenAttack.text_process.tokenizer import Tokenizer, PunctTokenizer
from OpenAttack.attack_assist.substitute.word import WordNetSubstitute
from OpenAttack.exceptions import WordNotInDictionaryException


class PWWSAttacker(SlowAttacker):
    def __init__(
        self, 
        device: Optional[torch.device] = None,
        tokenizer: Union[Tokenizer, BertTokenizerFast] = None,
        model: Union[BertForMaskedLM, BartForConditionalGeneration] = None,
        max_len: int = 64,
        max_per: int = 3,
        task: str = "seq2seq",
    ):
        super(PWWSAttacker, self).__init__(
            device, tokenizer, model, max_len, max_per, task,
        )
        self.unk_token = tokenizer.unk_token
        self.default_tokenizer = PunctTokenizer()
        self.substitute = WordNetSubstitute()
        self.filter_words = set(ENGLISH_FILTER_WORDS)

    def compute_loss(self, text: list, labels: list):
        return None, None

    @ torch.no_grad()
    def get_prediction(self, sentence: Union[str, List[str]]):
        return super().get_prediction(sentence)

    def mutation(
        self, 
        context: str, 
        sentence: str, 
        grad: torch.gradient, 
        goal: str, 
        modify_pos: List[int],
    ):
        new_strings = []
        modified_pos = set(modify_pos)
        x_orig = sentence.lower()
        x_orig = self.default_tokenizer.tokenize(x_orig)
        pos_tags =  list(map(lambda x: x[1], x_orig)) 
        x_orig =  list(map(lambda x: x[0], x_orig)) 
        # Word saliency
        S = self.get_saliency(context, x_orig, goal) # (len(sent), )
        S_softmax = np.exp(S - S.max())
        S_softmax = S_softmax / S_softmax.sum()
        # Substitutation saliency
        w_star = [self.get_wstar(context, x_orig, i, pos_tags[i], goal) for i in range(len(x_orig))] # [(rep_word, prob)]
        H = [(idx, w_star[idx][0], S_softmax[idx] * w_star[idx][1]) for idx in range(len(x_orig))] # (idx, rep_word, H_score)
        H = sorted(H, key=lambda x:-x[2])
        # print("H: {}".format(H))
        ret_sent = x_orig.copy()
        # Greedy perturbation
        for i in range(len(H)):
            idx, wd, _ = H[i]
            if ret_sent[idx] in self.filter_words or idx in modified_pos:
                continue
            ret_sent[idx] = wd # replace the word
            curr_sent = self.default_tokenizer.detokenize(ret_sent)
            new_strings.append((idx, curr_sent))

        return new_strings


    def get_saliency(self, context: str, sent: list, goal: str):
        x_hat_raw = []
        for i in range(len(sent)):
            left = sent[:i]
            right = sent[i + 1:]
            x_i_hat = left + [self.unk_token] + right
            x_i_hat = context + self.sp_token + self.default_tokenizer.detokenize(x_i_hat)
            x_hat_raw.append(x_i_hat)
        
        x_orig = context + self.sp_token + self.default_tokenizer.detokenize(sent)
        x_hat_raw.append(x_orig)
        scores, seqs, pred_len = self.compute_score(x_hat_raw, batch_size=5) # list N of [T X V], [T], [1]
        label = self.tokenizer(goal, truncation=True, max_length=self.max_len, return_tensors='pt')
        label = label['input_ids'][0] # (T, )
        res = self.get_target_p(scores, pred_len, label) # numpy array (N, )
        return res[-1] - res[:-1]
        

    def get_wstar(
        self, 
        context: str, 
        sent: list, 
        idx: int, 
        pos: str, 
        goal: str,
    ):
        word = sent[idx]
        try:
            rep_words = list(map(lambda x:x[0], self.substitute(word, pos)))
        except WordNotInDictionaryException:
            rep_words = []
        rep_words = list(filter(lambda x: x != word, rep_words))
        if len(rep_words) == 0:
            return (word, 0)
        sents = []
        for rw in rep_words:
            new_sent = sent[:idx] + [rw] + sent[idx + 1:]
            new_sent = context + self.sp_token + self.default_tokenizer.detokenize(new_sent)
            sents.append(new_sent)
        orig_sent = context + self.sp_token + self.default_tokenizer.detokenize(sent)
        sents.append(orig_sent)
        scores, seqs, pred_len = self.compute_score(sents, batch_size=5) # list of [T X V], [T], [1]
        label = self.tokenizer(goal, truncation=True, max_length=self.max_len, return_tensors='pt')
        label = label['input_ids'][0] # (T, )
        res = self.get_target_p(scores, pred_len, label) # numpy array (N, )
        prob_orig = res[-1]
        res = res[:-1]
        return (rep_words[res.argmin()], prob_orig - res.min())