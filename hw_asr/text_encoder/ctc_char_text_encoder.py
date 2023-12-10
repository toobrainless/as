from collections import defaultdict
from typing import List, NamedTuple

import numpy as np
import pyctcdecode
import torch

from hw_asr.utils.util import ROOT_PATH

from .char_text_encoder import CharTextEncoder
from .utils import ctc_beam_search as custom_ctc_beam_search


class Hypothesis(NamedTuple):
    text: str
    prob: float


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str], lm_params=None):
        super().__init__(alphabet)
        vocab = self.alphabet + [self.EMPTY_TOK]
        self.ind2char = dict(enumerate(vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}
        self.beam_decoder = pyctcdecode.build_ctcdecoder(self.alphabet)
        if lm_params is not None:
            with open(ROOT_PATH / lm_params["vocab_path"]) as f:
                unigram_list = [t.lower() for t in f.read().strip().split("\n")]
            self.beam_lm_decoder = pyctcdecode.build_ctcdecoder(
                self.alphabet,
                kenlm_model_path=str(ROOT_PATH / lm_params["kenlm_model_path"]),
                unigrams=unigram_list,
            )
        else:
            self.beam_lm_decoder = None

    def ctc_decode(self, inds: List[int]) -> str:
        ans = []
        last_token = self.EMPTY_TOK
        for ind in inds:
            cur_token = self.ind2char[ind]
            if cur_token == last_token:
                continue
            if cur_token == self.EMPTY_TOK:
                last_token = cur_token
                continue
            last_token = cur_token
            ans.append(cur_token)

        return "".join(ans)

    def ctc_argmax(self, log_probs: torch.tensor, probs_length) -> List[Hypothesis]:
        predictions = np.argmax(log_probs, axis=-1)

        return self.ctc_decode(predictions[:probs_length])

    def custom_ctc_beam_search(
        self, log_probs: torch.tensor, probs_length, beam_size: int = 4
    ) -> List[Hypothesis]:
        probs = np.exp(log_probs)[:probs_length]
        hypothesis = custom_ctc_beam_search(
            probs, beam_size, self.ind2char, self.EMPTY_TOK
        )

        return hypothesis[0][0]

    def pyctc_beam_search(
        self, log_probs: torch.tensor, probs_length, beam_size: int = 4
    ) -> List[Hypothesis]:
        return self.beam_decoder.decode(
            logits=log_probs[:probs_length], beam_width=beam_size
        )

    def ptctc_beam_search_lm(
        self, log_probs: torch.tensor, probs_length, beam_size: int = 4
    ) -> List[Hypothesis]:
        return self.beam_lm_decoder.decode(
            logits=log_probs[:probs_length], beam_width=beam_size
        )
