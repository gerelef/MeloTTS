import pickle
import os
import re
import sys

try:
    from . import symbols
    from .es_phonemizer import cleaner as es_cleaner
    from .es_phonemizer import es_to_ipa
except ImportError:
    import symbols
    from es_phonemizer import cleaner as es_cleaner
    from es_phonemizer import es_to_ipa

from transformers import AutoTokenizer, AutoModelForMaskedLM

# The original author for this module hopes that the following comments will serve
#  as a good educational / entry point to expanding the supported languages for MeloTTS, as
#  the Greek implementation is non-trivial, with respect to the reference (English).
# As such, I'll be documenting all the non-trivial steps performed in the two critical modules
# *greek.py*, *greek_bert.py* as much as possible. Starting off;
#
# Greek does not have a latin alphabet. Equivalent case (implementation) is the
#  Korean language.
# For the Korean language, the author(s) perform the following steps, in order:
#   - stripping for whitespace (left & right)
#   - deleting some sort of header - perhaps this was case specific, because
#      no such substitution existed in the equivalent spanish implementation.
#   - converting some korean to the equivalent english phrase.
#   - 'normalize' english to equivalent Korean word
#   - lowercase
#      AUTHOR'S NOTE: this is probably an optional step. This most likely has
#       to do with the 'BERT' model we're using later on; this module will also
#       use an 'uncased' (lowercase) tokenizer 'nlpaueb/bert-base-greek-uncased-v1'
#       https://huggingface.co/nlpaueb/bert-base-greek-uncased-v1

# @inproceedings{greek-bert,
# author = {Koutsikakis, John and Chalkidis, Ilias and Malakasiotis, Prodromos and Androutsopoulos, Ion},
# title = {GREEK-BERT: The Greeks Visiting Sesame Street},
# year = {2020},
# isbn = {9781450388788},
# publisher = {Association for Computing Machinery},
# address = {New York, NY, USA},
# url = {https://doi.org/10.1145/3411408.3411440},
# booktitle = {11th Hellenic Conference on Artificial Intelligence},
# pages = {110–117},
# numpages = {8},
# location = {Athens, Greece},
# series = {SETN 2020}
# }
#
# If you wish to experiment with another BERT model, you may export another model
#  as an environment variable.

_ENVVAR = 'GREEK_BERT_UNCASED_MODEL'
model_id = 'nlpaueb/bert-base-greek-uncased-v1'
# override the model_id with the provided model, if the mdl is NOT an empty string
if (mdl := os.environ[_ENVVAR] if _ENVVAR in os.environ else None) and mdl:
    print(f"OVERRIDING {model_id} with provided model {mdl} through envvar: {_ENVVAR}", file=sys.stderr)
    model_id = mdl

tokenizer = AutoTokenizer.from_pretrained(model_id)


def text_normalize(text: str) -> str:
    # As the `cleaner.py` for `spanish_cleaners` mentions, there is probably no
    #  need to expand abbreviation and numbers, as the phonemizer (g2p -- grapheme to phoneme)
    #  indeed already does that by itself (at least, it should -- for our model).

    return re.sub("(?![α-ωΑ-Ωά-ώΆ-Ώ .;:]+).", "",
                  text.replace("ς", "σ") \
                  .replace(";", "") \
                  .strip(),
                  flags=re.DOTALL)


greek_to_ipa = {
    'α': 'a', 'β': 'v', 'γ': 'ɣ', 'δ': 'ð', 'ε': 'e', 'ζ': 'z',
    'η': 'i', 'θ': 'θ', 'ι': 'i', 'κ': 'k', 'λ': 'l', 'μ': 'm',
    'ν': 'n', 'ξ': 'ks', 'ο': 'o', 'π': 'p', 'ρ': 'r', 'σ': 's',
    'τ': 't', 'υ': 'u', 'φ': 'f', 'χ': 'x', 'ψ': 'ps', 'ω': 'o',
    'ά': 'á', 'έ': 'é', 'ή': 'í', 'ί': 'í', 'ό': 'ó', 'ύ': 'ú',
    'ώ': 'ó',
    'αἰ': 'ai', 'αὐ': 'au', 'εἰ': 'ei', 'εὐ': 'eu', 'οἰ': 'oi',
    'οὐ': 'ou', 'υἱ': 'ui'
}


def _gr_to_ipa(word):
    result = ''
    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i:i + 2] in greek_to_ipa:
            result += greek_to_ipa[word[i:i + 2]]
            i += 2
        else:
            result += greek_to_ipa.get(word[i], word[i])
            i += 1
    return result


import unicodedata


def has_accents_or_diacritics(char):
    return len(unicodedata.normalize('NFD', char)) > 1


def distribute_phone(n_phone, n_word):
    phones_per_word = [0] * n_word
    for task in range(n_phone):
        min_tasks = min(phones_per_word)
        min_index = phones_per_word.index(min_tasks)
        phones_per_word[min_index] += 1
    return phones_per_word


def g2p(text, pad_start_end=True, tokenized=None):
    if tokenized is None:
        tokenized = tokenizer.tokenize(text_normalize(text))

    ph_groups = []
    for t in tokenized:
        if not t.startswith("#"):
            ph_groups.append([t])
        else:
            ph_groups[-1].append(t.replace("#", ""))

    phones = []
    tones = []
    word2ph = []
    for group in ph_groups:
        w = "".join(group)
        phone_len = 0
        word_len = len(group)
        if w == '[UNK]':
            phone_list = ['UNK']
        else:
            phone_list = list(filter(lambda p: p != " ", _gr_to_ipa(w)))

        for ph in phone_list:
            phones.append(ph)
            tones.append(0)
            phone_len += 1
        aaa = distribute_phone(phone_len, word_len)
        word2ph += aaa

    if pad_start_end:
        phones = ["_"] + phones + ["_"]
        tones = [0] + tones + [0]
        word2ph = [1] + word2ph + [1]
    return phones, tones, word2ph


def get_bert_feature(text, word2ph, device=None):
    try:
        from . import greek_bert
    except ImportError:
        import greek_bert

    return greek_bert.get_bert_feature(text, word2ph, device=device)


if __name__ == "__main__":
    for v in greek_to_ipa.values():
        print(f"\"{v}\"", end=', ')
    exit(1)
    from copy import copy

    og = "Ο παπάς ο παχύς έφαγε παχιά φακύ. Γιατί παπά παχύ έφαγες παχιά φακύ;"
    text = copy(og)
    # print(text)
    text = text_normalize(text)
    print(text)
    phones, tones, word2ph = g2p(text)
    bert = get_bert_feature(text, word2ph)

    print({
        "phones": {
            len(phones): phones
        },
        "tones": {len(tones): tones},
        len(word2ph): word2ph
    })
    print(sum(word2ph), bert.shape)
    print(og)
    print(''.join(text))
