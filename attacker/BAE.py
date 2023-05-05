# BAE: BERT-based Adversarial Examples for Text Classification
import copy 
import random
import numpy as np
from typing import Union, List, Optional
import torch
from transformers import (
    BertConfig, 
    BertTokenizerFast, 
    BertForMaskedLM,
    BartForConditionalGeneration,
)
from .base import SlowAttacker
from utils import ENGLISH_FILTER_WORDS, USE
from OpenAttack.data_manager import DataManager


class Feature(object):
    def __init__(self, seq_a, label):
        self.label = label
        self.seq = seq_a
        self.final_adverse = seq_a
        self.query = 0
        self.change = 0
        self.success = 0
        self.sim = 0.0
        self.changes = []


class BAEAttacker(SlowAttacker):
    def __init__(
        self,  
        device: Optional[torch.device] = None,
        tokenizer: Optional[BertTokenizerFast] = None,
        model: Optional[BartForConditionalGeneration] = None,
        max_len: int = 64,
        max_per: int = 3,
        task: str = 'seq2seq',
        replace_rate : float = 1.0, 
        insert_rate : float = 0.0, 
        batch_size : int = 32,
        threshold_pred_score: float = 0.3,
        ):
        super(BAEAttacker, self).__init__(
            device, tokenizer, model, max_len, max_per, task,
        )
        mlm_path = "bert-base-uncased"
        self.k = 50 # number of candidates
        self.filter_words = set(ENGLISH_FILTER_WORDS)
        self.use_encoder = USE()
        self.tokenizer_mlm = BertTokenizerFast.from_pretrained(mlm_path, do_lower_case=True)
        config_atk = BertConfig.from_pretrained(mlm_path)
        self.mlm_model = BertForMaskedLM.from_pretrained(mlm_path, config=config_atk).to(self.device)
        self.threshold_pred_score = threshold_pred_score # Threshold used in substitute module.
        self.max_length = max_len
        self.batch_size = batch_size
        self.replace_rate = replace_rate # Replace rate
        self.insert_rate = insert_rate # Insert rate

        if self.replace_rate == 1.0 and self.insert_rate == 0.0:
            self.sub_mode = 0 # only using replacement
        elif self.replace_rate == 0.0 and self.insert_rate == 1.0:
            self.sub_mode = 1 # only using insertion
        elif self.replace_rate + self.insert_rate == 1.0:
            self.sub_mode = 2 # replacement OR insertion for each token / subword
        elif self.replace_rate == 1.0 and self.insert_rate == 1.0:
            self.sub_mode = 3 # first replacement AND then insertion for each token / subword
        else:
            raise NotImplementedError()

    def compute_loss(self, text: list, labels: list):
        return None, None

    def _tokenize(self, seq: str):
        seq = seq.replace('\n', '').lower()
        words = seq.split(' ')
        sub_words = []
        keys = []
        index = 0
        for word in words:
            sub = self.tokenizer_mlm.tokenize(word) # handle subword
            sub_words += sub
            keys.append([index, index + len(sub)])
            index += len(sub)
        return words, sub_words, keys

    def _get_masked_insert(self, words: list):
        len_text = max(len(words), 2)
        masked_words = [] # list of words
        for i in range(len_text - 1):
            masked_words.append(words[0:i + 1] + ['[UNK]'] + words[i + 1:])
        return masked_words

    @ torch.no_grad()
    def get_prediction(self, sentence: Union[str, List[str]]):
        return super().get_prediction(sentence)

    
    def get_important_scores(
        self,
        context: str, 
        x_orig: str,
        words: list, 
        goal: str, 
    ):
        x_orig_text = context + self.sp_token + x_orig # add context
        masked_words = self._get_masked_insert(words)
        texts = [context + self.sp_token + ' '.join(words) for words in masked_words]  # list of masked texts
        texts.append(x_orig_text)
        scores, seqs, pred_len = self.compute_score(texts, batch_size=5) # list of [T X V], [T], [1]
        label = self.tokenizer(goal, truncation=True, max_length=self.max_len, return_tensors='pt')
        label = label['input_ids'][0] # (T, )
        res = self.get_target_p(scores, pred_len, label) # numpy array (T'+1, )
        return res[-1] - res[:-1]

    ##### TODO: make this one of the substitute unit under ./substitures #####
    def get_substitues(
        self, 
        masked_index: int, 
        tokens: list, 
        sub_mode: str,
    ):
        masked_tokens = copy.deepcopy(tokens)
        if sub_mode == "r":
            masked_tokens[masked_index] = '[MASK]'
        elif sub_mode == "i":
            masked_tokens.insert(masked_index, '[MASK]')
        else:
            raise NotImplementedError()
        
        # Convert token to vocabulary indices
        indexed_tokens = self.tokenizer_mlm.convert_tokens_to_ids(masked_tokens)
        # Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
        segments_ids = [0] * len(indexed_tokens)
        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens]).to(self.device)
        segments_tensors = torch.tensor([segments_ids]).to(self.device)
      
        self.mlm_model.eval()
        # Predict all tokens
        with torch.no_grad():
            outputs = self.mlm_model(tokens_tensor, token_type_ids=segments_tensors)
            predictions = outputs[0]

        predicted_indices = torch.topk(predictions[0, masked_index], self.k)[1]
        predicted_tokens = self.tokenizer_mlm.convert_ids_to_tokens(predicted_indices)
        return predicted_tokens
    

    def get_sim_embed(self, embed_path: str, sim_path: str):
        id2word = {}
        word2id = {}
        with open(embed_path, 'r', encoding='utf-8') as ifile:
            for line in ifile:
                word = line.split()[0]
                if word not in id2word:
                    id2word[len(id2word)] = word
                    word2id[word] = len(id2word) - 1

        cos_sim = np.load(sim_path)
        return cos_sim, word2id, id2word


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
        # MLM-process
        feature = Feature(x_orig, goal)
        words, sub_words, keys = self._tokenize(feature.seq)
        final_words = copy.deepcopy(words)
        max_length = self.max_length
        # Calculate the importance score
        important_scores = self.get_important_scores(context, x_orig, words, goal) # list
        feature.query += int(len(words))
        list_of_index = sorted(enumerate(important_scores), key=lambda x: x[1], reverse=True)

        sub_words = ['[CLS]'] + sub_words[:max_length - 2] + ['[SEP]'] # list of sub_words
        input_ids_ = torch.tensor([self.tokenizer_mlm.convert_tokens_to_ids(sub_words)]) # 1 X T
        word_predictions = self.mlm_model(input_ids_.to(self.device))[0].squeeze()  # T X V
        word_pred_scores_all, word_predictions = torch.topk(word_predictions, self.k, -1)  # T X k
        word_predictions = word_predictions[1:len(sub_words) + 1, :] # T' X k
        word_pred_scores_all = word_pred_scores_all[1:len(sub_words) + 1, :] # T' X k

        offset = 0
        for top_index in list_of_index: # (index, score)
            if feature.change > int(0.2 * (len(words))):
                feature.success = 1  # exceed
                return None
            tgt_word = words[top_index[0]]
            if tgt_word in self.filter_words:
                continue

            substitutes = word_predictions[keys[top_index[0]][0]:keys[top_index[0]][1]]  # |word| X k
            word_pred_scores = word_pred_scores_all[keys[top_index[0]][0]:keys[top_index[0]][1]]
            # In the substitute function, masked_index = top_index[0] + 1, 
            # because "[CLS]" has been inserted into sub_words
            replace_sub_len, insert_sub_len = 0, 0
            temp_sub_mode = -1
            # substitutes: list of candidate words
            if self.sub_mode == 0:
                substitutes = self.get_substitues(top_index[0] + 1, sub_words, 'r')
            elif self.sub_mode == 1:
                substitutes = self.get_substitues(top_index[0] + 1, sub_words, 'i')
            elif self.sub_mode == 2:
                rand_num = random.random()
                if rand_num < self.replace_rate:
                    substitutes = self.get_substitues(top_index[0] + 1, sub_words, 'r')
                    temp_sub_mode = 0
                else:
                    substitutes = self.get_substitues(top_index[0] + 1, sub_words, 'i')
                    temp_sub_mode = 1
            elif self.sub_mode == 3:
                substitutes_replace = self.get_substitues(top_index[0] + 1, sub_words, 'r') 
                substitutes_insert = self.get_substitues(top_index[0] + 1, sub_words, 'i') 
                replace_sub_len, insert_sub_len = len(substitutes_replace), len(substitutes_insert)
                substitutes = substitutes_replace + substitutes_insert
            else:
                raise NotImplementedError
            most_gap = 0.0
            candidate = None
            for i, substitute in enumerate(substitutes): # iterate all candidates
                if substitute == tgt_word:
                    continue  # filter out original word
                if '##' in substitute:
                    continue  # filter out sub-word
                if substitute in self.filter_words:
                    continue
                if self.sub_mode == 3:
                    if i < replace_sub_len:
                        temp_sub_mode = 0
                    else:
                        temp_sub_mode = 1
                temp_replace = copy.deepcopy(final_words)
                # Check if we should REPLACE or INSERT the substitute into the orignal word list 
                is_replace = self.sub_mode == 0 or temp_sub_mode == 0 
                is_insert = self.sub_mode == 1 or temp_sub_mode == 1 
                if is_replace:
                    orig_word = temp_replace[top_index[0]]
                    pos_tagger = DataManager.load("TProcess.NLTKPerceptronPosTagger")
                    pos_tag_list_before = [elem[1] for elem in pos_tagger(temp_replace)]
                    temp_replace[top_index[0]] = substitute
                    pos_tag_list_after = [elem[1] for elem in pos_tagger(temp_replace)]
                    # reverse temp_replace back to its original if pos_tag changes, and continue
                    # searching for the next best substitue
                    if pos_tag_list_after != pos_tag_list_before:
                        temp_replace[top_index[0]] = orig_word
                        continue
                elif is_insert:
                    temp_replace.insert(top_index[0] + offset, substitute)
                else:
                   raise NotImplementedError

                temp_text = self.tokenizer_mlm.convert_tokens_to_string(temp_replace)
                # use_score = self.encoder.calc_score(temp_text, x_orig)
                use_score = self.use_encoder.count_use(temp_text, x_orig)
                # From TextAttack's implementation: 
                # Finally, since the BAE code is based on the TextFooler code, we need to adjust 
                # the threshold to account for the missing / pi in the cosine similarity comparison. 
                # So the final threshold is 1 - (1 - 0.8) / pi = 1 - (0.2 / pi) = 0.936338023.
                if use_score < 0.936:
                    continue
                # scores, seqs, pred_len = self.compute_score([temp_text], batch_size=5) # list of [T X V], [T], [1]
                # temp_prob = scores[0].squeeze()
                # feature.query += 1
                # temp_prob = torch.softmax(temp_prob, -1)
                # temp_label = torch.argmax(temp_prob)
                # print("temp_label: ", temp_label)
                new_strings.append((top_index[0], temp_text))

            #     if goal.check(feature.final_adverse, temp_label):
            #         feature.change += 1
            #         if is_replace:
            #             final_words[top_index[0]] = substitute
            #         elif is_insert:
            #             final_words.insert(top_index[0] + offset, substitute)
            #         else:
            #             raise NotImplementedError()
            #         feature.changes.append([keys[top_index[0]][0], substitute, tgt_word])
            #         feature.final_adverse = temp_text
            #         feature.success = 4
            #         return feature.final_adverse
            #     else:
            #         label_prob = temp_prob[goal]
            #         gap = current_prob - label_prob
            #         if gap > most_gap:
            #             most_gap = gap
            #             candidate = substitute
            #     if is_insert:
            #         final_words.pop(top_index[0] + offset)

            # if most_gap > 0:
            #     feature.change += 1
            #     feature.changes.append([keys[top_index[0]][0], candidate, tgt_word])
            #     current_prob = current_prob - most_gap
            #     if is_replace:
            #         final_words[top_index[0]] = candidate
            #     elif is_insert:
            #         final_words.insert(top_index[0] + offset, candidate) 
            #         offset += 1
            #     else:
            #         raise NotImplementedError()
            
        # feature.final_adverse = (self.tokenizer_mlm.convert_tokens_to_string(final_words))
        # feature.success = 2
        # return None
        return new_strings
