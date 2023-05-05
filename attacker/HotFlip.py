from typing import Optional, List, Union
import numpy as np
import torch
from transformers import (
    BertTokenizerFast, 
    BertForMaskedLM,
    BartForConditionalGeneration,
)
from utils import ENGLISH_FILTER_WORDS
from .base import SlowAttacker
from OpenAttack.text_process.tokenizer import Tokenizer, PunctTokenizer
from OpenAttack.attack_assist.substitute.word import WordNetSubstitute
from OpenAttack.exceptions import WordNotInDictionaryException


class HotFlipAttacker(SlowAttacker):

    def __init__(
        self,
        device: Optional[torch.device] = None,
        tokenizer: Union[Tokenizer, BertTokenizerFast] = None,
        model: Union[BertForMaskedLM, BartForConditionalGeneration] = None,
        max_len: int = 64,
        max_per: int = 3,
        task: str = "seq2seq",
    ):
        super(HotFlipAttacker, self).__init__(
            device, tokenizer, model, max_len, max_per, task,
        )
        self.substitute = WordNetSubstitute()
        self.default_tokenizer = PunctTokenizer()
        self.filter_words = set(ENGLISH_FILTER_WORDS)
        
    def compute_loss(self, text: list, labels: list):
        return None, None
        
    def do_replace(self, x_cur, word, index):
        ret = x_cur
        ret[index] = word
        return ret
             
    def get_neighbours(self, word, POS):
        try:
            return list(map(lambda x: x[0], self.substitute(word, POS)))
        except WordNotInDictionaryException:
            return []

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
        x_pos =  list(map(lambda x: x[1], x_orig))
        x_orig = list(map(lambda x: x[0], x_orig))
        
        counter = -1
        for word, pos in zip(x_orig, x_pos):
            counter += 1
            if word in self.filter_words or counter in modified_pos:
                continue
            neighbours = self.get_neighbours(word, pos)
            for neighbour in neighbours:
                x_new = self.default_tokenizer.detokenize(self.do_replace(x_orig, neighbour, counter))
                if x_new != sentence:
                    new_strings.append((counter, x_new))
                
        return new_strings
      
    