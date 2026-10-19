"""Text-quality heuristics shared by extraction and dataset validation."""

import re

CYRILLIC = re.compile('[а-яёА-ЯЁўқғҳЎҚҒҲ]')
LATIN = re.compile('[a-zA-Z]')
JUNK_SYMBOLS = re.compile(r'[{}|*$^~<>\[\]\\]')


def fix_mojibake(text: str) -> str:
    """Repair cp1251 text that a PDF font mapped as cp1252 ("ÏËÀÑÒÈÍÛ" -> "ПЛАСТИНЫ")."""
    latin = len(re.findall("[À-ÿ]", text))
    if latin > 50 and latin > len(re.findall("[а-яА-ЯёЁ]", text)):
        try:
            return text.encode("cp1252").decode("cp1251")
        except UnicodeError:
            return text
    return text


def text_quality(text: str) -> float:
    """Share of visible characters that belong to words of 3+ letters.

    Readable documents score about 0.7-0.9. Broken text layers score low:
    glyph codes ("/G01/G02"), OCR noise ("lI1l)"), control characters.
    Not meaningful for short answers such as "01.07.81".
    """
    sample = text[:30000]
    visible = len(re.sub(r"\s", "", sample))
    if not visible:
        return 0.0
    return sum(len(word) for word in re.findall(r"[^\W\d_]{3,}", sample)) / visible


def noise_ratio(text: str) -> float:
    """Share of 3+ char tokens that look like OCR noise.

    A noisy token mixes Cyrillic and Latin letters ("Мехжш,t,нн") or has junk
    symbols next to letters ("к}рикдан"). Readable text, numbers and lists of
    standards ("ГОСТ 12.1.004-91") score near 0.
    """
    tokens = [token for token in text.split() if len(token) >= 3]
    if not tokens:
        return 0.0
    noisy = 0
    for token in tokens:
        core = token.strip('.,;:!?()«»"\'-–—')
        mixed = CYRILLIC.search(core) and LATIN.search(core)
        junk = JUNK_SYMBOLS.search(core) and re.search(r'[^\W\d_]', core)
        noisy += bool(mixed or junk)
    return noisy / len(tokens)
