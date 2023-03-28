# This lib doesn't have a working sentence tokeniser
# https://github.com/alvations/sacremoses/tree/master/sacremoses

from typing import Dict, List
from mosestokenizer import *

cacheSplitters: Dict[str, any] = {}

# for use in this file only.


def ssplit(textString: str, lang_2letter: str) -> List[str]:

    if (lang_2letter not in cacheSplitters):
        cacheSplitters[lang_2letter] = MosesSentenceSplitter(lang_2letter)

    splitter = cacheSplitters[lang_2letter]

    # This also filters out empty lines, otherwise MosesSentenceSplitter says:
    # "blank lines are not allowed"
    return splitter([s for s in textString.splitlines() if s])
