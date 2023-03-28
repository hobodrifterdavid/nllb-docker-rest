from typing import Dict
from stopes_snippet.ssplit_stopes import get_split_algo
from sacremoses import MosesPunctNormalizer
import re
import xxhash
from stopes_snippet.remove_regex import get_non_printing_char_replacer
import unicodedata

cacheSplitters: Dict[str, any] = {}


def splitAndClean(lang_flores: str, text: str):

    if (lang_flores not in cacheSplitters):
        cacheSplitters[lang_flores] = SentenceSplitClean(
            lang_flores, split_algo="default")

    splitter = cacheSplitters[lang_flores]

    results = list(splitter(text))
    sentences = [s for (h, s, c) in results]
    cleaned = [c for (h, s, c) in results]

    return cleaned


class SentenceSplitClean:
    def __init__(self, splitter_lang: str, split_algo: str):
        # setup sentence splitter
        self.splitter = get_split_algo(splitter_lang, split_algo=split_algo)

        # setup "moses" normalization
        self.mpn = MosesPunctNormalizer(lang="en")
        self.mpn.substitutions = [
            (re.compile(r), sub) for r, sub in self.mpn.substitutions
        ]
        self.replace_nonprint = get_non_printing_char_replacer(
            " ")

    def __call__(self, line):
        sentence_splits = self.splitter(line)
        line_hash = xxhash.xxh3_64_intdigest(line)

        for sent in sentence_splits:
            # normalize -- moses equivalent
            clean = self.mpn.normalize(sent)
            clean = self.replace_nonprint(clean)
            # replace 𝓕𝔯𝔞𝔫𝔠𝔢𝔰𝔠𝔞 by Francesca
            clean = unicodedata.normalize("NFKC", clean)

            yield (line_hash, sent, clean)
