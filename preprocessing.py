"""Preprocessing bersama untuk inferensi dan evaluasi teks Indonesia."""

import re
from pathlib import Path

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


STOP_WORDS = frozenset(
    (Path(__file__).resolve().parent / "data" / "stopwords_indonesian.txt")
    .read_text(encoding="utf-8")
    .splitlines()
)
_stemmer = StemmerFactory().create_stemmer()


def preprocess(text):
    """Bersihkan, stem, lalu hapus stopword sesuai pipeline penelitian."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = _stemmer.stem(text)
    return " ".join(word for word in text.split() if word not in STOP_WORDS)
