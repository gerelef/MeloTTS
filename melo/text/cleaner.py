from . import chinese, japanese, english, chinese_mix, korean, french, spanish, greek
from . import cleaned_text_to_sequence
import copy
import sys

# Implementees must support the following three interfaces (functions):
# - text_normalize(text)
# - g2p(norm_text)
# - get_bert_feature(norm_text, word2ph, device=device)
#
# These dependencies are documented in the current file.

language_module_map = {
    "ZH": chinese,
    "JP": japanese,
    "EN": english,
    "ZH_MIX_EN": chinese_mix,
    "KR": korean,
    "FR": french,
    "SP": spanish,
    "ES": spanish,
    "GR" : greek
}


def clean_text(text, language):
    # sanity check
    if language not in language_module_map:
        print(f"Cannot match {language} to any preprocessing step; please provide the language code in uppercase!", file=sys.stderr)
        print(f"Current supported language codes are: {language_module_map.keys()}", file=sys.stderr)
        exit(2)

    language_module = language_module_map[language]
    norm_text = language_module.text_normalize(text)
    phones, tones, word2ph = language_module.g2p(norm_text)

    return norm_text, phones, tones, word2ph


def clean_text_bert(text, language, device=None):
    # sanity check
    if language not in language_module_map:
        print(f"Cannot match {language} to any preprocessing step; please provide the language code in uppercase!", file=sys.stderr)
        print(f"Current supported language codes are: {language_module_map.keys()}", file=sys.stderr)
        exit(2)

    language_module = language_module_map[language]
    norm_text = language_module.text_normalize(text)
    phones, tones, word2ph = language_module.g2p(norm_text)

    word2ph_bak = copy.deepcopy(word2ph)
    for i in range(len(word2ph)):
        word2ph[i] = word2ph[i] * 2
    word2ph[0] += 1
    bert = language_module.get_bert_feature(norm_text, word2ph, device=device)

    return norm_text, phones, tones, word2ph_bak, bert


def text_to_sequence(text, language):
    # sanity check
    if language not in language_module_map:
        print(f"Cannot match {language} to any preprocessing step; please provide the language code in uppercase!", file=sys.stderr)
        print(f"Current supported language codes are: {language_module_map.keys()}", file=sys.stderr)
        exit(2)

    norm_text, phones, tones, word2ph = clean_text(text, language)
    return cleaned_text_to_sequence(phones, tones, language)


if __name__ == "__main__":
    pass
