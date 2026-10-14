import csv
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import tqdm


LAB_DIR = Path(__file__).parent
PROJECT_ROOT = LAB_DIR.parents[1]
TOKENS_PHONES_DATA = LAB_DIR / 'data' / 'tokens_phones.csv'
MFA_DICTIONARY = LAB_DIR / 'data' / 'russian_mfa.dict'
TARGET_PHONES = PROJECT_ROOT / 'common' / 'phones_mfa.txt'

SIL = '<SIL>'
VOWELS = set('аеёиоуыэюя')


PHONE_FAMILIES = {
    'а': {'a', 'ɐ', 'ə'},
    'б': {'b', 'bʲ', 'p', 'pʲ'},
    'в': {'v', 'vʲ', 'f', 'fʲ'},
    'г': {'ɡ', 'ɟ', 'k', 'x', 'ç', 'ɣ'},
    'д': {'d̪', 'dʲ', 't̪', 'tʲ', 'dʐ'},
    'е': {'e', 'ɛ', 'ɪ', 'ə', 'j'},
    'ё': {'o', 'ɵ', 'j'},
    'ж': {'ʐ', 'ʑ', 'ʂ', 'dʐ'},
    'з': {'z̪', 'zʲ', 's̪', 'sʲ'},
    'и': {'i', 'ɪ', 'ɨ'},
    'й': {'j'},
    'к': {'k', 'c', 'ɡ', 'ɟ'},
    'л': {'ɫ', 'ʎ'},
    'м': {'m', 'mʲ'},
    'н': {'n̪', 'ɲ'},
    'о': {'o', 'ɐ', 'ə'},
    'п': {'p', 'pʲ', 'b', 'bʲ'},
    'р': {'r', 'rʲ'},
    'с': {'s̪', 'sʲ', 'z̪', 'zʲ', 'ɕ'},
    'т': {'t̪', 'tʲ', 'd̪', 'dʲ', 't̪s̪', 'tɕ'},
    'у': {'u', 'ʊ', 'ʉ'},
    'ф': {'f', 'fʲ', 'v', 'vʲ'},
    'х': {'x', 'ç'},
    'ц': {'t̪s̪'},
    'ч': {'tɕ'},
    'ш': {'ʂ', 'ɕ'},
    'щ': {'ɕ'},
    'ъ': set(),
    'ы': {'ɨ'},
    'ь': set(),
    'э': {'ɛ', 'e'},
    'ю': {'u', 'ʉ', 'j'},
    'я': {'a', 'æ', 'ɪ', 'ə', 'j'},
}


def normalize_token(token):
    return unicodedata.normalize('NFC', str(token)).lower()


def load_mfa_dictionary(path=MFA_DICTIONARY):
    lexicon = defaultdict(Counter)
    with open(path, encoding='utf-8') as file:
        for line in file:
            parts = line.rstrip().split('\t')
            if len(parts) < 2:
                continue
            word = normalize_token(parts[0])
            phones = parts[-1].replace('ː', '')
            if phones and 'spn' not in phones.split():
                lexicon[word][phones] += 1
    return dict(lexicon)


def alignment_cost(char, phones):
    family = PHONE_FAMILIES.get(char, set())
    if not phones:
        return 0 if char in 'ъь' else 2.2
    if all(phone in family for phone in phones):
        return 0.05 * (len(phones) - 1)
    matched = sum(phone in family for phone in phones)
    return 1.7 * (len(phones) - matched) + 0.25 * matched


def align_word(word, transcription):
    phones = transcription.split()
    costs = np.full((len(word) + 1, len(phones) + 1), np.inf)
    back = {}
    costs[0, 0] = 0

    for i, char in enumerate(word):
        for j in range(len(phones) + 1):
            if not np.isfinite(costs[i, j]):
                continue
            for size in range(3):
                end = j + size
                phones_left = len(phones) - end
                chars_left = len(word) - i - 1
                if end > len(phones) or phones_left > chars_left * 2:
                    continue
                fragment = tuple(phones[j:end])
                cost = costs[i, j] + alignment_cost(char, fragment)
                if cost < costs[i + 1, end]:
                    costs[i + 1, end] = cost
                    back[i + 1, end] = size

    if not np.isfinite(costs[-1, -1]):
        return [tuple()] * len(word)

    result = []
    j = len(phones)
    for i in range(len(word), 0, -1):
        size = back[i, j]
        result.append(tuple(phones[j - size:j]))
        j -= size
    return result[::-1]


def get_features(word, i):
    padded = '^^' + word + '$$'
    j = i + 2
    window = padded[j - 2:j + 3]
    vowels_left = sum(char in VOWELS for char in word[i:])
    suffix = word[-4:]
    prev_char, char, next_char = padded[j - 1:j + 2]
    return [
        (window, suffix, vowels_left),
        (window, vowels_left),
        (prev_char, char, next_char, suffix[-2:], vowels_left),
        (prev_char, char, next_char, vowels_left),
        (prev_char, char, next_char),
        (char,),
    ]


class G2P:
    def __init__(self, lexicon=None, mfa_lexicon=None):
        self.lexicon = lexicon or {}
        self.mfa_lexicon = mfa_lexicon or load_mfa_dictionary()
        self.model = [defaultdict(Counter) for _ in range(6)]

        with open(TARGET_PHONES, encoding='utf-8') as file:
            self.target_phones = {line.strip() for line in file if line.strip()}

        for word, variants in self.lexicon.items():
            if not word or any(char not in PHONE_FAMILIES for char in word):
                continue
            phones, frequency = variants.most_common(1)[0]
            for i, output in enumerate(align_word(word, phones)):
                for level, feature in enumerate(get_features(word, i)):
                    self.model[level][feature][output] += min(frequency, 5)

    def _lookup(self, token, lexicon):
        variants = lexicon.get(token)
        if variants:
            return variants.most_common(1)[0][0].split()

    def _predict_word(self, word):
        result = []
        for i in range(len(word)):
            for level, feature in enumerate(get_features(word, i)):
                variants = self.model[level].get(feature)
                if variants:
                    result.extend(variants.most_common(1)[0][0])
                    break
        return result

    def transcribe(self, tokens: list[str] | np.ndarray) -> list[list[str]]:
        result = []
        for token in tokens:
            if token == SIL:
                result.append([SIL])
                continue

            word = normalize_token(token)
            phones = self._lookup(word, self.lexicon)
            phones = phones or self._lookup(word, self.mfa_lexicon)
            phones = phones or self._predict_word(word)

            unknown = set(phones) - self.target_phones
            if unknown:
                raise ValueError(f'unknown phones for {token}: {sorted(unknown)}')
            result.append(phones)
        return result


def phone_error_rate(reference: list[list[str]], hypothesis: list[list[str]]) -> float:
    total_distance = 0
    total_length = 0
    for ref, hyp in zip(reference, hypothesis):
        total_length += len(ref)
        if not hyp:
            total_distance += len(ref)
            continue
        previous = list(range(len(hyp) + 1))
        for i, ref_phone in enumerate(ref, start=1):
            current = [i] + [0] * len(hyp)
            for j, hyp_phone in enumerate(hyp, start=1):
                current[j] = min(previous[j] + 1,
                                 current[j - 1] + 1,
                                 previous[j - 1] + (ref_phone != hyp_phone))
            previous = current
        total_distance += previous[-1]
    return total_distance / total_length if total_length else float('nan')


def word_accuracy(reference: list[list[str]], hypothesis: list[list[str]]) -> float:
    if not reference:
        return float('nan')
    return float(np.mean([ref == hyp for ref, hyp in zip(reference, hypothesis)]))


def load_tokens(path=TOKENS_PHONES_DATA):
    tokens = pd.read_csv(path, sep='|', quoting=csv.QUOTE_NONE)
    tokens['token'] = tokens.token.astype(str).map(normalize_token)
    tokens['phones'] = tokens.phones.fillna('').astype(str)
    bad = (tokens.phones == '') | tokens.phones.str.split().map(lambda p: 'spn' in p)
    return tokens[~bad].reset_index(drop=True)


def build_lexicon(tokens):
    lexicon = defaultdict(Counter)
    for token, phones in tokens[['token', 'phones']].values:
        lexicon[token][phones] += 1
    return dict(lexicon)


def calc_metrics(g2p, tokens, corpus_lexicon, mfa_lexicon):
    groups = {name: ([], []) for name in ['text', 'corpus', 'mfa', 'model']}

    for _, sentence in tqdm.tqdm(tokens.groupby('id', sort=False)):
        sentence = sentence.sort_values('token_id')
        words = sentence.token.values
        predicted = g2p.transcribe(words)
        reference = [phones.split() for phones in sentence.phones]

        groups['text'][0].append([phone for word in reference for phone in word])
        groups['text'][1].append([phone for word in predicted for phone in word])

        for token, ref, hyp in zip(words, reference, predicted):
            if token in corpus_lexicon:
                group = 'corpus'
            elif token in mfa_lexicon:
                group = 'mfa'
            else:
                group = 'model'
            groups[group][0].append(ref)
            groups[group][1].append(hyp)

    metrics = {}
    for name, (reference, hypothesis) in groups.items():
        metrics[name] = {
            'tokens': len(reference),
            'per': phone_error_rate(reference, hypothesis),
            'word_accuracy': word_accuracy(reference, hypothesis),
        }
    return metrics


def test_g2p():
    tokens = load_tokens()
    train = tokens[tokens.set == 'train']
    test = tokens[tokens.set == 'test']
    lexicon = build_lexicon(train)
    mfa_lexicon = load_mfa_dictionary()
    g2p = G2P(lexicon, mfa_lexicon)
    metrics = calc_metrics(g2p, test, lexicon, mfa_lexicon)
    print(pd.DataFrame(metrics).T.to_string(float_format=lambda value: f'{value:.4f}'))
    return metrics


if __name__ == '__main__':
    test_g2p()
