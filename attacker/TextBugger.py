from typing import Optional, List, Union
import random
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



class TextBuggerAttacker(SlowAttacker):

    def __init__(
        self,
        device: Optional[torch.device] = None,
        tokenizer: Union[Tokenizer, BertTokenizerFast] = None,
        model: Union[BertForMaskedLM, BartForConditionalGeneration] = None,
        max_len: int = 64,
        max_per: int = 3,
        task: str = "seq2seq",
        ):
        super(TextBuggerAttacker, self).__init__(
            device, tokenizer, model, max_len, max_per, task,
        )
        self.substitute = WordNetSubstitute()
        self.default_tokenizer = PunctTokenizer()
        self.filter_words = set(ENGLISH_FILTER_WORDS)
        
        
    def bug_sub_W(self, word: str):
        try:
            res = self.substitute(word, None)
            if len(res) == 0:
                return word
            return res[0][0]
        except WordNotInDictionaryException:
            return word

    def bug_insert(self, word: str):
        if len(word) >= 6:
            return word
        res = word
        point = random.randint(1, len(word) - 1)
        res = res[0:point] + " " + res[point:]
        return res

    def bug_delete(self, word: str):
        res = word
        point = random.randint(1, len(word) - 2)
        res = res[0:point] + res[point + 1:]
        return res

    def bug_swap(self, word: str):
        if len(word) <= 4:
            return word
        res = word
        points = random.sample(range(1, len(word) - 1), 2)
        a = points[0]
        b = points[1]
        res = list(res)
        w = res[a]
        res[a] = res[b]
        res[b] = w
        res = ''.join(res)
        return res

    def bug_sub_C(self, word: str):
        res = word
        key_neighbors = self.get_key_neighbors()
        point = random.randint(0, len(word) - 1)

        if word[point] not in key_neighbors:
            return word
        choices = key_neighbors[word[point]]
        subbed_choice = choices[random.randint(0, len(choices) - 1)]
        res = list(res)
        res[point] = subbed_choice
        res = ''.join(res)
        return res

    def get_key_neighbors(self):
        # By keyboard proximity
        neighbors = {
            "q": "was", "w": "qeasd", "e": "wrsdf", "r": "etdfg", "t": "ryfgh", "y": "tughj", "u": "yihjk",
            "i": "uojkl", "o": "ipkl", "p": "ol",
            "a": "qwszx", "s": "qweadzx", "d": "wersfxc", "f": "ertdgcv", "g": "rtyfhvb", "h": "tyugjbn",
            "j": "yuihknm", "k": "uiojlm", "l": "opk",
            "z": "asx", "x": "sdzc", "c": "dfxv", "v": "fgcb", "b": "ghvn", "n": "hjbm", "m": "jkn"
        }
        # By visual proximity
        neighbors['i'] += '1'
        neighbors['l'] += '1'
        neighbors['z'] += '2'
        neighbors['e'] += '3'
        neighbors['a'] += '4'
        neighbors['s'] += '5'
        neighbors['g'] += '6'
        neighbors['b'] += '8'
        neighbors['g'] += '9'
        neighbors['q'] += '9'
        neighbors['o'] += '0'
        return neighbors
        
        
    def compute_loss(self, text: list, labels: list):
        cls_loss = self.get_cls_loss(text, labels)
        return (None, cls_loss)
    
    
    def get_w_word_importances(self, sentence_tokens: List[str], grad: torch.Tensor): # white  
        gradient = grad.detach().cpu().numpy()
        sent_grad = []
        for idx, tok in enumerate(sentence_tokens):
            orig_id = self.tokenizer.convert_tokens_to_ids(tok)
            sent_grad.append(gradient[orig_id])
        sent_grad = np.array(sent_grad)
        if sent_grad.shape[0] != len(sentence_tokens):
            raise RuntimeError("Sent %d != Gradient %d" % (len(sentence_tokens), sent_grad.shape[0]))
        dist = np.linalg.norm(sent_grad, axis=1)
        return [idx for idx, _ in sorted(enumerate(dist.tolist()), key=lambda x: -x[1])]
    
    
    def generateBugs(self, word: str):
        bugs = {"insert": word, "delete": word, "swap": word, "sub_C": word, "sub_W": word}
        if len(word) <= 2:
            return bugs
        bugs["insert"] = self.bug_insert(word)
        bugs["delete"] = self.bug_delete(word)
        bugs["swap"] = self.bug_swap(word)
        bugs["sub_C"] = self.bug_sub_C(word)
        bugs["sub_W"] = self.bug_sub_W(word)
        return bugs
    
    def replaceWithBug(self, x_prime: List[str], word_idx: int, bug: str):
        return x_prime[:word_idx] + [bug] + x_prime[word_idx + 1:]
        
    
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
        x = self.default_tokenizer.tokenize(sentence, pos_tagging=False)
        ranked_words = self.get_w_word_importances(x, grad) # list of word index
        for word_idx in ranked_words:
            word = x[word_idx]
            if word in self.filter_words or word_idx in modified_pos:
                continue
            # bug = self.selectBug(context, word, word_idx, x, goal) # bug word
            # x = self.replaceWithBug(x, word_idx, bug) # replace orig sentence
            bugs = self.generateBugs(word) # dict {bug type: bug word}
            for bug_type, b_k in bugs.items():
                candidate_k = self.replaceWithBug(x, word_idx, b_k) # list of words in a sentence
                x_prime_sentence = self.default_tokenizer.detokenize(candidate_k)
                if x_prime_sentence != sentence:
                    new_strings.append((word_idx, x_prime_sentence))

        return new_strings


    
    