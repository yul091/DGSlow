
import numpy as np
from typing import Optional, List, Union
import torch
from transformers import (
    BertTokenizerFast, 
    BertForMaskedLM,
    BartForConditionalGeneration,
)
from .base import SlowAttacker
from utils import ENGLISH_FILTER_WORDS
from OpenAttack.text_process.tokenizer import Tokenizer, PunctTokenizer
from OpenAttack.attack_assist.substitute.word import WordNetSubstitute
from OpenAttack.exceptions import WordNotInDictionaryException


class FDAttacker(SlowAttacker):
    def __init__(
        self,
        device: Optional[torch.device] = None,
        tokenizer: Union[Tokenizer, BertTokenizerFast] = None,
        model: Union[BertForMaskedLM, BartForConditionalGeneration] = None,
        max_len: int = 64,
        max_per: int = 3,
        task: str = "seq2seq",
    ):
        super(FDAttacker, self).__init__(
            device, tokenizer, model, max_len, max_per, task,
        )
        self.substitute = WordNetSubstitute()
        self.default_tokenizer = PunctTokenizer()
        self.filter_words = set(ENGLISH_FILTER_WORDS)
        self.unk_token = tokenizer.unk_token
        
        
    def compute_loss(self, text: list, labels: list):
        cls_loss = self.get_cls_loss(text, labels)
        return (None, cls_loss)
    
    
    def mutation(
        self, 
        context: str, 
        sentence: str, 
        grad: torch.gradient, 
        goal: str, 
        modify_pos: List[int],
    ):
        new_strings = []
        x_orig = sentence.lower()
        sent = self.default_tokenizer.tokenize(x_orig, pos_tagging=False)
        avail_pos = list(set(range(len(sent))) - set(modify_pos))
        if not avail_pos:
            return []
         
        for i in range(50):
            iter_cnt = 0
            while True:
                # idx = np.random.choice(len(sent)) # randomly choose a word
                idx = np.random.choice(avail_pos)
                iter_cnt += 1
                if iter_cnt > 5 * len(sent): # failed to find a substitute word
                    return []
                if sent[idx] in self.filter_words:
                    continue
                try: # find a substitute word
                    reps = list(map(lambda x:x[0], self.substitute(sent[idx], None)))
                except WordNotInDictionaryException:
                    continue
                reps = list(filter(lambda x: self.tokenizer.convert_tokens_to_ids(x) != self.unk_token, reps))
                if len(reps) > 0:
                    break
            
            orig_id = self.tokenizer.convert_tokens_to_ids(sent[idx])
            gradient = grad.detach().cpu().numpy()
            # embedding = self.embedding.detach().cpu().numpy()
            s1 = np.sign(gradient[orig_id]) # (E, )
            mn = None
            mnwd = None
            
            for word in reps:
                word_id = self.tokenizer.convert_tokens_to_ids(word)
                # s0 = np.sign(embedding[word_id] - embedding[orig_id]) # (E, )
                s0 = np.sign(gradient[word_id] - gradient[orig_id]) # (E, )
                v = np.abs(s0 - s1).sum()
                
                if (mn is None) or v < mn:
                    mn = v
                    mnwd = word

            if mnwd is not None:
                sent[idx] = mnwd
                curr_sent = self.default_tokenizer.detokenize(sent)
                new_strings.append((idx, curr_sent))
        
        return new_strings
    