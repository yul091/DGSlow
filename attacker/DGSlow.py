import torch
import torch.nn.functional as F
import nltk
from transformers import (
    AutoTokenizer, 
    BertTokenizer,
    AutoModelForMaskedLM,
    BartForConditionalGeneration,
)
import stanza
from typing import Union, List
from nltk.corpus import wordnet as wn
from utils import GrammarChecker, ENGLISH_FILTER_WORDS, DEFAULT_TEMPLATES
from .base import SlowAttacker



class WordAttacker(SlowAttacker):
    def __init__(
        self, 
        device: torch.device = None,
        tokenizer: BertTokenizer = None,
        model: BartForConditionalGeneration = None,
        max_len: int = 64,
        max_per: int = 3,
        task: str = 'seq2seq',
        fitness: str = 'length',
        select_beams: int = 1,
    ):
        super(WordAttacker, self).__init__(
            device, tokenizer, model, max_len, 
            max_per, task, fitness, select_beams,
        )
        self.num_of_perturb = 50
        self.filter_words = set(ENGLISH_FILTER_WORDS)


    def compute_loss(self, text: list, labels: list):
        scores, seqs, pred_len = self.compute_score(text) # list of [T X V], [T], [1]
        # loss_list = self.leave_eos_target_loss(scores, seqs, pred_len)
        loss_list = self.leave_eos_loss(scores, pred_len)
        return (loss_list, None)
    

    def token_replace_mutation(
        self, 
        cur_adv_text: str, 
        grad: torch.gradient, 
        modified_pos: List[int],
    ):
        new_strings = []
        # words = self.tokenizer.tokenize(cur_adv_text)
        cur_inputs = self.tokenizer(cur_adv_text, return_tensors="pt", padding=True)
        cur_ids = cur_inputs.input_ids[0].to(self.device)
        base_ids = cur_ids.clone()
        # current_text = self.tokenizer.decode(cur_ids, skip_special_tokens=True)
        for pos, t in enumerate(cur_ids):
            if t not in self.special_id and pos not in modified_pos:
                cnt, grad_t = 0, grad[t]
                score = (self.embedding - self.embedding[t]).mm(grad_t.reshape([-1, 1])).reshape([-1])
                index = score.argsort()
                for tgt_t in index:
                    if tgt_t not in self.special_token:
                        new_base_ids = base_ids.clone()
                        new_base_ids[pos] = tgt_t
                        candidate_s = self.tokenizer.decode(new_base_ids, skip_special_tokens=True)
                        # if new_tag[pos][:2] == ori_tag[pos][:2]:
                        new_strings.append((pos, candidate_s))
                        cnt += 1
                        if cnt >= 10:
                            break

        return new_strings

    def mutation(
        self, 
        context: str, 
        cur_adv_text: str, 
        grad: torch.gradient, 
        label: str, 
        modified_pos: List[int],
    ):
        new_strings = self.token_replace_mutation(cur_adv_text, grad, modified_pos)
        # print('new strings: ', new_strings)
        return new_strings



class StructureAttacker(SlowAttacker):
    def __init__(
        self, 
        device: torch.device = None,
        tokenizer: BertTokenizer = None,
        model: BartForConditionalGeneration = None,
        max_len: int = 64,
        max_per: int = 3,
        task: str = 'seq2seq',
        fitness: str = 'length',
        select_beams: int = 1,
        eos_weight: float = 0.5,
        cls_weight: float = 0.5,
        delta: float = 0.5,
        use_combined_loss: bool = False,
    ):
        super(StructureAttacker, self).__init__(
            device, tokenizer, model, max_len, max_per, task, fitness,
            select_beams, eos_weight, cls_weight, delta, use_combined_loss,
        )
        self.filter_words = set(ENGLISH_FILTER_WORDS)
        # BERT initialization
        self.berttokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.mask_token = self.berttokenizer.mask_token
        bertmodel = AutoModelForMaskedLM.from_pretrained('bert-base-uncased')
        self.bertmodel = bertmodel.eval().to(self.device)
        self.num_of_perturb = 50
        self.grammar = GrammarChecker()
        self.pos_dict = {'NOUN': 'n', 'VERB': 'v', 'ADV': 'r', 'ADJ': 'a'}
        self.pos_processor = stanza.Pipeline('en', processors='tokenize, mwt, pos, lemma')
        self.templates = DEFAULT_TEMPLATES
        # self.skip_pos_tags = ['DT', 'PDT', 'POS', 'PRP', 'PRP$', 'TO', 'WDT', 'WP', 'WP$', 'WRB', 'NNP']

    def get_pos(self, sentence: str, mask_index: int):
        processed_sentence = self.pos_processor(sentence)
        pos_list = []
        word_lemma = None

        for sentence in processed_sentence.sentences:
            for i, word in enumerate(sentence.words):
                pos_list.append(word.upos)
                if i == mask_index:
                    word_lemma = word.lemma

        return pos_list, word_lemma

    @staticmethod
    def get_word_antonyms(word: str):
        antonyms_lists = set()
        for syn in wn.synsets(word):
            for l in syn.lemmas():
                if l.antonyms():
                    antonyms_lists.add(l.antonyms()[0].name())
        return list(antonyms_lists)

    def get_synonyms(self, word: str, pos: str):
        if word is None:
            return []
        if pos not in self.pos_dict.keys():
            return []
        synonyms = set()
        for syn in wn.synsets(word):
            if syn.pos() == self.pos_dict[pos]:
                for lemma in syn.lemmas():
                    synonyms.add(lemma.name())
        if word in synonyms:
            synonyms.remove(word)
        return list(synonyms)

    def formalize(self, text: Union[str, List[str]]):
        tokens = self.berttokenizer.tokenize(text)
        if len(tokens) > self.max_len:
            tokens = tokens[0:self.max_len]

        string = self.berttokenizer.convert_tokens_to_string(tokens)
        return string

    def compute_loss(self, text: list, labels: list):
        scores, seqs, pred_len = self.compute_score(text)
        loss_list = self.leave_eos_target_loss(scores, seqs, pred_len)
        # loss_list = self.leave_eos_loss(scores, pred_len)
        cls_loss = self.get_cls_loss(text, labels)
        return (loss_list, cls_loss)

    def get_token_type(self, input_tensor: torch.Tensor):
        # tokens = self.tree_tokenizer.tokenize(sent)
        tokens = self.tokenizer.convert_ids_to_tokens(input_tensor)
        pos_inf = nltk.tag.pos_tag(tokens)
        bert_masked_indexL = list()
        # collect the token index for substitution
        for idx, (word, tag) in enumerate(pos_inf):
            # substitute the nouns and adjectives; you could easily substitue more words by modifying the code here
            # if tag.startswith('NN') or tag.startswith('JJ'):
            #     tagFlag = tag[:2]
                # we do not perturb the first and the last token because BERT's performance drops on for those positions
            # if idx != 0 and idx != len(tokens) - 1:
            bert_masked_indexL.append((idx, tag))

        return tokens, bert_masked_indexL


    def BertSubstitute(
        self, 
        cur_text: str, 
        cur_tokens: list, 
        cur_tags: list, 
        cur_error: Union[float, int], 
        masked_index: int,
    ):
        new_sentences = []
        # invalidChars = set(string.punctuation)
        # For each idx, use BERT to generate k (i.e., num) candidate tokens
        cur_tok = cur_tokens[masked_index]
        low_tokens = [x.lower() for x in cur_tokens]
        low_tokens[masked_index] = self.mask_token

        # Get the pos tag & synonyms of the masked word
        # pos_list, word_lemma = self.get_pos(cur_text, masked_index)
        # masked_word_pos = pos_list[masked_index]
        # synonyms = self.get_synonyms(word_lemma, masked_word_pos)
        antonyms = self.get_word_antonyms(cur_tok)
        # print("antonyms: ", antonyms)

        # Try whether all the tokens are in the vocabulary
        try:
            indexed_tokens = self.berttokenizer.convert_tokens_to_ids(low_tokens)
            tokens_tensor = torch.tensor([indexed_tokens]).to(self.device)
            prediction = self.bertmodel(tokens_tensor)[0]
            # Skip the sentences that contain unknown words
            # Another option is to mark the unknow words as [MASK]; 
            # we skip sentences to reduce fp caused by BERT
        except KeyError as error:
            print('skip a sentence. unknown token is %s' % error)
            return new_sentences

        # Get the similar words
        probs = F.softmax(prediction[0, masked_index], dim=-1)
        topk_Idx = torch.topk(probs, self.num_of_perturb, sorted=True)[1].tolist()
        topk_tokens = self.berttokenizer.convert_ids_to_tokens(topk_Idx)
        
        # Handle subtokens
        def handle_subtokens(tokens: list):
            new_tokens = []
            for tok in tokens:
                if tok.startswith('##') and new_tokens:
                    new_tokens[-1] = new_tokens[-1] + tok[2:]
                else:
                    new_tokens.append(tok)
            return new_tokens
        
        topk_tokens = handle_subtokens(topk_tokens)
        # Remove the tokens that only contains 0 or 1 char (e.g., i, a, s)
        # This step could be further optimized by filtering more tokens (e.g., non-english tokens)
        topk_tokens = list(filter(lambda x: len(x) > 1, topk_tokens))
        topk_tokens = list(set(topk_tokens) - set(self.berttokenizer.all_special_tokens) - set(antonyms))
        # topk_tokens = list(topk_tokens | set(synonyms)) # union with WordNet synonyms
        # print("topk_tokens: ", topk_tokens)
        # Generate similar sentences
        for tok in topk_tokens:
            # if any(char in invalidChars for char in tok):
            #     continue
            cur_tokens[masked_index] = ' '+tok
            new_pos_inf = nltk.tag.pos_tag(cur_tokens)
            # Only use sentences whose similar token's tag is still the same
            if new_pos_inf[masked_index][1] == cur_tags[masked_index][1]:
                # print("[index {}] substituted: {}, pos tag: {}".format(
                #     masked_index, tok, new_pos_inf[masked_index][1],
                # ))
                # new_t = self.tokenizer.encode(tokens[masked_index], add_special_tokens=False)[0]
                # new_tensor = ori_tensors.clone()
                # new_tensor[masked_index] = new_t
                # new_sentence = self.tokenizer.decode(new_tensor, skip_special_tokens=True)
                # new_sentence = self.formalize(new_sentence)
                new_sentence = self.tokenizer.convert_tokens_to_string(cur_tokens)
                # print("new sentence: ", new_sentence)
                new_error = self.grammar.check(new_sentence)
                # if new_error > cur_error:
                #     continue
                new_sentences.append((masked_index, new_sentence))

        cur_tokens[masked_index] = cur_tok
        return new_sentences


    def structure_mutation(
        self, 
        cur_adv_text: str, 
        grad: torch.gradient, 
        modify_pos: List[int],
    ):
        """
        cur_adv_text (string): the current adversarial text;
        grad (tensor[V X E]): the gradient of the current adversarial text.
        """
        all_new_strings = []
        modified_pos = set(modify_pos)
        important_tensor = (-grad.sum(1)).argsort() # sort token ids w.r.t. gradient
        important_tokens = self.tokenizer.convert_ids_to_tokens(important_tensor.tolist())
        cur_input = self.tokenizer(cur_adv_text, return_tensors="pt", add_special_tokens=False)
        cur_tensor = cur_input['input_ids'][0]
        cur_tokens, cur_tags = self.get_token_type(cur_tensor)
        cur_error = self.grammar.check(cur_adv_text)
        assert len(cur_tokens) == len(cur_tensor)
        assert len(cur_tokens) == len(cur_tags)

        def removeBPE(word: str):
            if word.startswith('▁'):
                return word.lstrip('▁').lower()
            if word.startswith('Ġ'):
                return word.lstrip('Ġ').lower()
            return word.lower()

        # For each important token (w.r.t. gradient), perturb it using BERT
        # if it is in the current text
        for tok in important_tokens:
            if (tok not in cur_tokens) or (removeBPE(tok) in self.filter_words):
                continue
            pos_list = [i for i, x in enumerate(cur_tokens) if x == tok]
            # print("\ncurrent key token: {}, pos tag: {}".format(tok, cur_tags[pos_list[0]][1]))
            for pos in pos_list:
                # if (cur_tags[pos][1] not in self.skip_pos_tags) and (pos not in modified_pos):
                if pos not in modified_pos:
                    new_strings = self.BertSubstitute(cur_adv_text, cur_tokens, cur_tags, cur_error, pos)
                    all_new_strings.extend(new_strings)
            if len(all_new_strings) > 2000:
                break

        return all_new_strings

    def mutation(
        self, 
        context: str, 
        cur_adv_text: str, 
        grad: torch.gradient, 
        label: str, 
        modified_pos: List[int],
    ):
        new_strings = self.structure_mutation(cur_adv_text, grad, modified_pos)
        return new_strings




