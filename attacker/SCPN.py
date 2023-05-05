from typing import Optional, Union, List
import torch
import pickle
import numpy as np
from .base import SlowAttacker
from utils import DEFAULT_TEMPLATES
from model.scpn import SCPN, ParseNet
from OpenAttack.text_process.tokenizer import Tokenizer, PunctTokenizer
from OpenAttack.text_process.constituency_parser import StanfordParser
from OpenAttack.data_manager import DataManager
from OpenAttack.attackers.scpn.subword import read_vocabulary, BPE



def reverse_bpe(sent):
    x = []
    cache = ''
    for w in sent:
        if w.endswith('@@'):
            cache += w.replace('@@', '')
        elif cache != '':
            x.append(cache + w)
            cache = ''
        else:
            x.append(w)
    return ' '.join(x)


class SCPNAttacker(SlowAttacker):

    def __init__(
        self,
        device: Optional[torch.device] = None,
        tokenizer: Optional[Tokenizer] = None,
        model: Optional[torch.nn.Module] = None,
        max_len: int = 64,
        max_per: int = 3,
        task: str = "seq2seq",
    ):
        super(SCPNAttacker, self).__init__(
            device, tokenizer, model, max_len, max_per, task,
        )
        self.templates = DEFAULT_TEMPLATES
        self.default_tokenizer = PunctTokenizer()
        self.parser = StanfordParser()

        # Use DataManager Here
        model_path = DataManager.load("AttackAssist.SCPN")
        pp_model = torch.load(model_path["scpn.pt"], map_location=self.device)
        parse_model = torch.load(model_path["parse_generator.pt"], map_location=self.device)
        pp_vocab, rev_pp_vocab = pickle.load(open(model_path["parse_vocab.pkl"], 'rb'))
        bpe_codes = open(model_path["bpe.codes"], "r", encoding="utf-8")
        bpe_vocab = open(model_path["vocab.txt"], "r", encoding="utf-8")
        self.parse_gen_voc = pickle.load(open(model_path["ptb_tagset.pkl"], "rb"))

        self.pp_vocab = pp_vocab
        self.rev_pp_vocab = rev_pp_vocab
        self.rev_label_voc = dict((v,k) for (k,v) in self.parse_gen_voc.items())

        # load paraphrase network
        pp_args = pp_model['config_args']
        self.net = SCPN(pp_args.d_word, pp_args.d_hid, pp_args.d_nt, pp_args.d_trans, len(self.pp_vocab), len(self.parse_gen_voc) - 1, pp_args.use_input_parse)
        self.net.load_state_dict(pp_model['state_dict'])
        self.net = self.net.to(self.device).eval()

        # load parse generator network
        parse_args = parse_model['config_args']
        self.parse_net = ParseNet(parse_args.d_nt, parse_args.d_hid, len(self.parse_gen_voc))
        self.parse_net.load_state_dict(parse_model['state_dict'])
        self.parse_net = self.parse_net.to(self.device).eval()

        # instantiate BPE segmenter
        bpe_vocab = read_vocabulary(bpe_vocab, 50)
        self.bpe = BPE(bpe_codes, '@@', bpe_vocab, None)


    def gen_paraphrase(self, sent: str, templates: List[str]):
        template_lens = [len(x.split()) for x in templates]
        np_templates = np.zeros((len(templates), max(template_lens)), dtype='int32')
        for z, template in enumerate(templates):
            np_templates[z, :template_lens[z]] = [self.parse_gen_voc[w] for w in template.split()]
        tp_templates = torch.from_numpy(np_templates).long().to(self.device)
        tp_template_lens = torch.LongTensor(template_lens).to(self.device)

        ssent =  ' '.join(self.default_tokenizer.tokenize(sent, pos_tagging=False))
        seg_sent = self.bpe.segment(ssent.lower()).split()
        
        # encode sentence using pp_vocab, leave one word for EOS
        seg_sent = [self.pp_vocab[w] for w in seg_sent if w in self.pp_vocab]

        # add EOS
        seg_sent.append(self.pp_vocab['EOS'])
        torch_sent = torch.LongTensor(seg_sent).to(self.device)
        torch_sent_len = torch.LongTensor([len(seg_sent)]).to(self.device)

        # encode parse using parse vocab
        # Stanford Parser
        parse_tree = self.parser(sent)
        parse_tree = " ".join(parse_tree.replace("\n", " ").split()).replace("(", "( ").replace(")", " )")
        parse_tree = parse_tree.split()

        for i in range(len(parse_tree) - 1):
            if (parse_tree[i] not in "()") and (parse_tree[i + 1] not in "()"):
                parse_tree[i + 1] = ""
        parse_tree = " ".join(parse_tree).split() + ["EOP"]

        torch_parse = torch.LongTensor([self.parse_gen_voc[w] for w in parse_tree]).to(self.device)
        torch_parse_len = torch.LongTensor([len(parse_tree)]).to(self.device)

        # generate full parses from templates
        beam_dict = self.parse_net.batch_beam_search(torch_parse.unsqueeze(0), tp_templates, torch_parse_len[:], tp_template_lens, self.parse_gen_voc['EOP'], beam_size=3, max_steps=150)
        seq_lens = []
        seqs = []
        for b_idx in beam_dict:
            prob,_,_,seq = beam_dict[b_idx][0]
            seq = seq[:-1] # chop off EOP
            seq_lens.append(len(seq))
            seqs.append(seq)
        np_parses = np.zeros((len(seqs), max(seq_lens)), dtype='int32')
        for z, seq in enumerate(seqs):
            np_parses[z, :seq_lens[z]] = seq
        tp_parses = torch.from_numpy(np_parses).long().to(self.device)
        tp_len = torch.LongTensor(seq_lens).to(self.device)

        # generate paraphrases from parses
        ret = []
        beam_dict = self.net.batch_beam_search(torch_sent.unsqueeze(0), tp_parses, torch_sent_len[:], tp_len, self.pp_vocab['EOS'], beam_size=3, max_steps=40)
        for b_idx in beam_dict:
            prob,_,_,seq = beam_dict[b_idx][0]
            gen_parse = ' '.join([self.rev_label_voc[z] for z in seqs[b_idx]])
            gen_sent = ' '.join([self.rev_pp_vocab[w] for w in seq[:-1]])
            ret.append(reverse_bpe(gen_sent.split()))
        return ret

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
        try:
            pps = self.gen_paraphrase(sentence, self.templates)
        except KeyError as e:
            return None
        # preds = victim.get_pred(pps)
        new_strings = [(0, p) for p in pps]
        return new_strings


