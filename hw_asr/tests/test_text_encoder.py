import unittest

from hw_asr.text_encoder.ctc_char_text_encoder import CTCCharTextEncoder


class TestTextEncoder(unittest.TestCase):
    def test_ctc_decode(self):
        alphabet = list(" abcdefghijklmnopqrstuvwxyz")
        text_encoder = CTCCharTextEncoder(alphabet)
        texts = [
            "i^^ ^w^i^sss^hhh^   i ^^^s^t^aaaar^teee^d dddddd^oooo^in^g tttttttth^iiiis h^^^^^^^^w^ e^a^r^li^er",
            "^^^^^hhhhhheeeelllll^lo ^^^^^^^w^^^^or^^^ld",
        ]
        true_texts = ["i wish i started doing this hw earlier", "hello world"]
        for text, true_text in zip(texts, true_texts):
            inds = [text_encoder.char2ind[c] for c in text]
            decoded_text = text_encoder.ctc_decode(inds)
            self.assertEqual(decoded_text, true_text)

    def test_beam_search(self):
        # TODO: (optional) write tests for beam search
        pass
