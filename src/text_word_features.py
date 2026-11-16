"""Word-level caption features for the text model (notebooks/06_text_model.ipynb).

One row per word of each caption. Words are defined by `text.split()`, the same way `fake_text_pos` indexes
them, so word indices line up with the edited-word labels.

Every feature is a fixed function of the caption (VADER lexicon, frozen GPT-2, spaCy part-of-speech tags).
As with 03's features, they are therefore safe to compute the same way on all splits. Anything learned from the data
(word frequencies, the word classifier) belongs in the notebook, fitted on train only.
"""

import numpy as np
import pandas as pd
import torch
from spacy.tokens import Doc

POS_TAGS = ["ADJ", "ADV", "VERB", "NOUN", "PROPN"]  # every other tag counts as "other"


@torch.no_grad()
def word_surprise(captions, lm, tokenizer, device, batch_size=64):
    """GPT-2 surprise of every word: the summed negative log-likelihood of its sub-word tokens.

    Each word is tokenised on its own (with its leading space), so every token maps back to exactly one word.
    As in 03, a BOS token is prepended so the first word is also scored. Returns one array per caption.
    """
    out = []
    for start in range(0, len(captions), batch_size):
        token_ids, owners = [], []
        for caption in captions[start:start + batch_size]:
            ids, owner = [tokenizer.bos_token_id], [-1]
            for i, word in enumerate(caption.split()):
                pieces = tokenizer.encode(word if i == 0 else " " + word)
                ids += pieces
                owner += [i] * len(pieces)
            token_ids.append(ids)
            owners.append(owner)

        # Right-padded batch: padding comes after the caption, so causal attention never sees it
        width = max(len(ids) for ids in token_ids)
        input_ids = torch.full((len(token_ids), width), tokenizer.eos_token_id)
        mask = torch.zeros((len(token_ids), width), dtype=torch.long)
        for r, ids in enumerate(token_ids):
            input_ids[r, :len(ids)] = torch.tensor(ids)
            mask[r, :len(ids)] = 1

        logits = lm(input_ids=input_ids.to(device), attention_mask=mask.to(device)).logits[:, :-1]
        nll = torch.nn.functional.cross_entropy(logits.transpose(1, 2), input_ids[:, 1:].to(device),
                                                reduction="none").float().cpu().numpy()

        # nll[r, j] is the surprise of token j + 1, which belongs to word owners[r][j + 1]
        for r, owner in enumerate(owners):
            per_word = np.zeros(owner[-1] + 1)
            np.add.at(per_word, np.array(owner[1:]), nll[r, :len(owner) - 1])
            out.append(per_word)
    return out


def pos_tags(captions, nlp, batch_size=512):
    """Universal part-of-speech tag of every word. spaCy is given the words as-is, so tags stay aligned with
    `text.split()` instead of spaCy's own tokenisation."""
    docs = (Doc(nlp.vocab, words=caption.split()) for caption in captions)
    return [[token.pos_ for token in doc] for doc in nlp.pipe(docs, batch_size=batch_size)]


def word_table(captions, lexicon, nlp, lm, tokenizer, device):
    """One row per (caption, word): `text`, `word_idx`, `word`, plus the `w_*` features.

    - `w_valence`, `w_abs_valence`: VADER valence of the word (0 if not in the lexicon)
    - `w_contrast`: positive when the word's sentiment is opposite to the rest of the caption, as when one
      sentiment word is flipped
    - `w_neighbour_abs_valence`: strongest valence of the words just before and after
    - `w_nll`, `w_nll_vs_caption`, `w_nll_is_max`: GPT-2 surprise of the word, relative to the caption's
      average, and whether it is the caption's most surprising word
    - `w_rel_position`, `w_len`, `w_is_capitalised`, `w_pos_<TAG>`: position, length, capitalisation and
      part of speech
    """
    captions = list(captions)
    surprise = word_surprise(captions, lm, tokenizer, device)
    tags = pos_tags(captions, nlp)

    columns = {name: [] for name in ["text", "word_idx", "word", "w_valence", "w_abs_valence", "w_contrast",
                                     "w_neighbour_abs_valence", "w_nll", "w_nll_vs_caption", "w_nll_is_max",
                                     "w_rel_position", "w_len", "w_is_capitalised", "pos"]}
    for caption, nll, tag in zip(captions, surprise, tags):
        words = caption.split()
        n = len(words)
        valence = np.array([lexicon.get(w.lower(), 0.0) for w in words])
        abs_valence = np.abs(valence)
        padded = np.pad(abs_valence, 1)

        columns["text"] += [caption] * n
        columns["word_idx"] += list(range(n))
        columns["word"] += words
        columns["w_valence"] += list(valence)
        columns["w_abs_valence"] += list(abs_valence)
        columns["w_contrast"] += list(-valence * (valence.sum() - valence))
        columns["w_neighbour_abs_valence"] += list(np.maximum(padded[:-2], padded[2:]))
        columns["w_nll"] += list(nll)
        columns["w_nll_vs_caption"] += list(nll - nll.mean())
        columns["w_nll_is_max"] += list((nll == nll.max()).astype(int))
        columns["w_rel_position"] += list(np.arange(n) / max(n - 1, 1))
        columns["w_len"] += [len(w) for w in words]
        columns["w_is_capitalised"] += [int(w[:1].isupper()) for w in words]
        columns["pos"] += tag

    table = pd.DataFrame(columns)
    for t in POS_TAGS:
        table[f"w_pos_{t}"] = (table["pos"] == t).astype("int8")
    return table.drop(columns="pos")
