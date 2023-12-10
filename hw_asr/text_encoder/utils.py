from collections import defaultdict
from typing import List, NamedTuple


class Hypothesis(NamedTuple):
    text: str
    prob: float


def extend_and_merge(frame, state, ind2char, empty_tok):
    new_state = defaultdict(float)
    for next_char_index, next_char_proba in enumerate(frame):
        for (pred, last_char), pref_proba in state.items():
            next_char = ind2char[next_char_index]
            if next_char == last_char:
                new_pref = pred
            else:
                if next_char != empty_tok:
                    new_pref = pred + next_char
                else:
                    new_pref = pred
                last_char = next_char
            new_state[(new_pref, last_char)] += pref_proba * next_char_proba
    return new_state


def truncate(state, beam_size):
    state_list = list(state.items())
    state_list.sort(key=lambda x: -x[1])
    return dict(state_list[:beam_size])


def ctc_beam_search(probs, beam_size, ind2char, empty_tok="^"):
    state = {("", empty_tok): 1.0}
    for frame in probs:
        state = extend_and_merge(frame, state, ind2char, empty_tok)
        state = truncate(state, beam_size)
    state_list = list(state.items())
    state_list.sort(key=lambda x: -x[1])
    return [Hypothesis(v[0][0], v[-1]) for v in state_list]
