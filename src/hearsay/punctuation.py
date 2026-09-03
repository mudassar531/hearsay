"""Sentence and clause boundary marks, across scripts.

The paragraph grouper and the clip segmenter both score a boundary higher when the
text before it ends a sentence or a clause. Both used to test for ASCII marks only,
so an Urdu ``۔``, a Hindi ``।``, an Arabic ``؟`` or a Chinese ``。`` never counted,
and exactly the languages hearsay is measured on were cut on pauses and duration
alone. One table here, imported by both, so the two can never drift apart again.
"""

import re

# Closing quotes and brackets a mark may be followed by, including the CJK corner
# brackets Whisper emits for quoted Japanese and Chinese speech.
_CLOSERS = "\"“”‘’')\\]」』》〉】"

_SENTENCE_MARKS = (
    ".!?"  # Latin, Cyrillic, Greek, Vietnamese, Turkish ...
    "。！？"  # CJK fullwidth stop, exclamation, question
    "؟"  # Arabic question mark — Arabic, Persian, Urdu, Pashto, Sindhi
    "۔"  # Urdu full stop
    "।॥"  # Devanagari danda / double danda — Hindi, Marathi, Nepali, Bengali, Punjabi
    "።"  # Ethiopic full stop — Amharic, Tigrinya
    "။"  # Myanmar
    "។៕"  # Khmer
    "།"  # Tibetan shad
    "։"  # Armenian full stop
)
_CLAUSE_MARKS = (
    ",;:—–"  # Latin and friends, plus the dashes
    "、，：；"  # CJK enumeration comma, fullwidth comma, colon, semicolon
    "،؛"  # Arabic comma, Arabic semicolon
    "՝"  # Armenian comma
)

# An ellipsis ending: a soft trailing-off, scored between clause and sentence.
ELLIPSIS_END = re.compile(rf"(?:\.\.\.|…)[{re.escape(_CLOSERS)}]*$")
# A sentence-final mark, optionally followed by closing quotes/brackets.
SENTENCE_END = re.compile(rf"[{re.escape(_SENTENCE_MARKS)}][{re.escape(_CLOSERS)}]*$")
# A clause-final mark (weaker, but still better than mid-phrase).
CLAUSE_END = re.compile(rf"[{re.escape(_CLAUSE_MARKS)}][{re.escape(_CLOSERS)}]*$")
