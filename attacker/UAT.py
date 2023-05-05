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


class UATAttacker(SlowAttacker):
    def __init__(
        self,
        device: Optional[torch.device] = None,
        tokenizer: Union[Tokenizer, BertTokenizerFast] = None,
        model: Union[BertForMaskedLM, BartForConditionalGeneration] = None,
        max_len: int = 64,
        max_per: int = 3,
        task: str = "seq2seq",
        ):
        super(UATAttacker, self).__init__(
            device, tokenizer, model, max_len, max_per, task,
        )
        self.substitute = WordNetSubstitute()
        self.default_tokenizer = PunctTokenizer()
        self.filter_words = set(ENGLISH_FILTER_WORDS)
        self.unk_token = tokenizer.unk_token
        self.beam_size = 50
        self.triggers = ["the" for _ in range(max_per)]
        
    
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
        trigger_len = len(self.triggers)
        modified_pos = set(modify_pos)
        
        orig_sent = self.default_tokenizer.tokenize(sentence, pos_tagging=False)
        if modify_pos:
            orig_sent = orig_sent[trigger_len:]
        
        def removeBPE(word: str):
            if word.startswith('▁'):
                return word.lstrip('▁').lower()
            if word.startswith('Ġ'):
                return word.lstrip('Ġ').lower()
            return word.lower()
        
        important_tensor = (-grad.sum(1)).argsort()[:self.beam_size] # sort token ids w.r.t. gradient
        important_tokens = self.tokenizer.convert_ids_to_tokens(important_tensor.tolist())
        
        for i in range(trigger_len):
            if i in modified_pos:
                continue
            for cw in important_tokens:
                cw = removeBPE(cw)
                tt = self.triggers[:i] + [cw] + self.triggers[i + 1:]
                xt = self.default_tokenizer.detokenize(tt + orig_sent)
                new_strings.append((i, xt))
        
        return new_strings