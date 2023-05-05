import random 
import numpy as np 
from typing import Optional, Union, List
import torch
from .base import SlowAttacker
from OpenAttack.text_process.tokenizer import Tokenizer
from OpenAttack.attack_assist.substitute.char import DCESSubstitute, ECESSubstitute

DEFAULT_CONFIG = {
    "prob": 0.3,
    "topn": 12,
    "generations": 120,
    "eces": True
}

class VIPERAttacker(SlowAttacker):

    def __init__(
        self,
        device: Optional[torch.device] = None,
        tokenizer: Optional[Tokenizer] = None,
        model: Optional[torch.nn.Module] = None,
        max_len: int = 64,
        max_per: int = 3,
        prob : float = 0.3,
        topn : int = 12,
        generations : int = 120,
        method: str = "eces",
        task: str = "seq2seq",
    ):
        super(VIPERAttacker, self).__init__(
            device, tokenizer, model, max_len, max_per, task,
        )
        self.prob = prob
        self.topn = topn
        self.generations = generations
        self.method = method
        self.sim_dict = {}

        if method == "dces":
            self.substitute = DCESSubstitute()
        elif method == "eces":
            self.substitute = ECESSubstitute()
        else:
            raise ValueError("Unknown method `%s` expect `%s`" % (method, ["dces", "eces"]))

    def compute_loss(self, text: list, labels: list):
        return None, None

    def mutation(
        self, 
        context: str, 
        sentence: str, 
        grad: torch.gradient, 
        goal: str, 
        modify_pos: List[int],
    ):
        new_strings = []
        for _ in range(self.generations):
            out = []
            for c in sentence:
                if self.method == "dces":
                    if c not in self.sim_dict:
                        similar_chars, probs = [], []
                        dces_list = self.substitute(c)[:self.topn]
                        for sc, pr in dces_list:
                            similar_chars.append(sc)
                            probs.append(pr)
                        probs = probs / np.sum(probs)
                        self.sim_dict[c] = (similar_chars, probs)
                    else:
                        similar_chars, probs = self.sim_dict[c]

                    r = random.random()
                    if r < self.prob and len(similar_chars) > 0:
                        s = np.random.choice(similar_chars, 1, replace=True, p=probs)[0]
                    else:
                        s = c
                    out.append(s)
                else:
                    r = random.random()
                    if r < self.prob:
                        s = self.substitute(c)[0][0]
                    else:
                        s = c
                    out.append(s)
            new_string = "".join(out)
            new_strings.append((0, new_string))

        return new_strings