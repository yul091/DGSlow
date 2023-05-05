import copy
import torch
from utils import (
    GrammarChecker,
    ConstituencyParser,
    T5,
)
from .base import SlowAttacker
from typing import Optional, Union, List
from transformers import (
    BertTokenizer,
    BertForMaskedLM,
    BartForConditionalGeneration,
)

class MAYAAttacker(SlowAttacker):
    def __init__(
        self,
        device: Optional[torch.device] = None,
        tokenizer: BertTokenizer = None,
        model: Union[BertForMaskedLM, BartForConditionalGeneration] = None,
        max_len: int = 64,
        max_per: int = 3,
        task: str = "seq2seq",
    ):
        super(MAYAAttacker, self).__init__(
            device, tokenizer, model, max_len, max_per, task,
        )
        self.parser = ConstituencyParser()
        self.grammar = GrammarChecker()
        self.paraphraser = T5()


    def compute_loss(self, text: list, labels: list):
        return None, None

    def get_masked_sentence(self, sentence:str):
        pos_info = None
        words = sentence.split(' ')
        masked_sentences = []
        indices = []
        for i in range(len(words)):
            word = words[i]
            words[i] = '[MASK]'
            tgt = ' '.join(x for x in words)
            masked_sentences.append(tgt)
            words[i] = word
            indices.append(i)

        return masked_sentences, pos_info, indices


    def get_best_sentences(self, sentence, paraphrases, info):
        ori_error = self.grammar.check(sentence)
        best_advs = []
        new_info = []

        for i in range(len(paraphrases[0])):
            advs = []
            for types in paraphrases:
                if types[i] is None:
                    continue

                adv_error = self.grammar.check(types[i])
                if adv_error <= ori_error:
                    advs.append(types[i])

            if len(advs) == 0:
                continue
            elif len(advs) == 1:
                best_advs.append(advs[0])
            else:
                best_adv = self.sent_encoder.find_best_sim(sentence, advs)[0]
                best_advs.append(best_adv)

            new_info.append(info[i])

        return best_advs, new_info


    def mutation(
        self, 
        context: str, 
        sentence: str, 
        grad: torch.gradient, 
        goal: str, 
        modify_pos: List[int],
    ):
        masked_sentences, word_info, masked_indices = self.get_masked_sentence(sentence)
        masked_new_strings = []
        for i in range(len(masked_indices)):
            masked_new_strings.append((masked_indices[i], masked_sentences[i]))

        root, nodes = self.parser(sentence)
        if len(nodes) == 0:
            return []

        phrases = [node[1] for node in nodes if node[3]]
        indices = [node[2] for node in nodes if node[3]]
        info = [[node[1], node[3], node[4]] for node in nodes]
        paraphrases = []
        with torch.no_grad():
            if phrases:
                one_batch = self.paraphraser.paraphrase(phrases)
                if one_batch is not None:
                    paraphrases.append(one_batch)

        translated_sentence_list = []
        if len(paraphrases) > 0:
            for paraphrase_list in paraphrases:
                translated_sentences = []
                for i, phrase in enumerate(paraphrase_list):
                    tree = self.parser.get_tree(phrase)
                    try:
                        root_copy = copy.deepcopy(root)
                        root_copy[indices[i]] = tree[0]
                        modified_sentence = ' '.join(word for word in root_copy.leaves())
                        translated_sentences.append(modified_sentence)

                    except Exception as e:
                        translated_sentences.append(None)

                translated_sentence_list.append(translated_sentences)

        best = []
        if len(translated_sentence_list) > 0:
            try:
                best, info = self.get_best_sentences(sentence, translated_sentence_list, info)

            except Exception as e:
                for i in translated_sentence_list:
                    print(len(i))
                print('error in getting best paraphrases!')
                best = []
                
        ori_words = sentence.split(' ')
        modified_words = [each_sentence.split(' ') for each_sentence in best]
        paraphrased_indices = []
        for i in range(len(modified_words)):
            indices = []
            if len(modified_words[i]) == len(ori_words):
                for j in range(len(ori_words)):
                    if modified_words[i][j] != ori_words[j]:
                        indices.append(j)
                paraphrased_indices.append(indices)
            elif len(modified_words[i]) < len(ori_words):
                for j in range(len(modified_words[i])):
                    if modified_words[i][j] != ori_words[j]:
                        indices.append(j)
                indices = indices + list(range(len(modified_words[i]), len(ori_words)))
                paraphrased_indices.append(indices)
            else:
                for j in range(len(ori_words)):
                    if modified_words[i][j] != ori_words[j]:
                        indices.append(j)
                paraphrased_indices.append(indices)
                
        paraphrased_new_strings = []
        for i in range(len(paraphrased_indices)):
            paraphrased_new_strings.append((paraphrased_indices[i], best[i]))
        
        new_strings = masked_new_strings + paraphrased_new_strings
        return new_strings

