"""Rule-based keyword extraction for memory records."""

import re
from typing import Optional


_STOP_WORDS = frozenset([
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "can", "of", "in", "on", "at", "to",
    "for", "from", "by", "with", "as", "or", "and", "but", "if", "not",
    "no", "yes", "it", "its", "this", "that", "these", "those", "i", "you",
    "he", "she", "we", "they", "what", "which", "who", "when", "where", "why",
    "how", "all", "each", "every", "both", "few", "more", "most", "other",
    "some", "such", "no", "nor", "only", "own", "same", "so", "than", "too",
    "very", "just", "also", "well", "now", "then", "here", "there", "up",
    "down", "out", "over", "under", "about", "after", "before", "between",
    "through", "during", "above", "below", "into", "onto", "off", "against",
    "along", "among", "around", "behind", "beside", "beyond", "inside",
    "outside", "without", "within", "across", "along", "among", "around",
    "之", "了", "的", "是", "在", "和", "有", "一", "个", "这", "那",
    "我", "你", "他", "她", "它", "们", "我们", "你们", "他们", "她们",
])


def extract_keywords(text: str, max_keywords: int = 20) -> list[str]:
    """Extract keywords from text using rule-based approach.

    Attempts to use RAGFlow's term_weight module if available, falls back to
    simple regex-based extraction.

    Args:
        text: Input text to extract keywords from.
        max_keywords: Maximum number of keywords to return.

    Returns:
        List of extracted keywords, sorted by relevance.
    """
    if not text or not isinstance(text, str):
        return []

    try:
        # Try using RAGFlow's existing term_weight module
        from rag.nlp import rag_tokenizer, term_weight

        tokens = rag_tokenizer.tokenize(text.lower()).split()
        if tokens:
            tw = term_weight.Dealer()
            weighted = tw.weights(tokens, preprocess=False)
            keywords = [
                t for t, _ in sorted(weighted, key=lambda x: -x[1])
                if t not in _STOP_WORDS and len(t) >= 2
            ][:max_keywords]
            return keywords
    except Exception:
        pass

    # Fallback: simple regex-based extraction
    return _extract_keywords_fallback(text, max_keywords)


def _extract_keywords_fallback(text: str, max_keywords: int = 20) -> list[str]:
    """Fallback keyword extraction using regex word splitting.

    Args:
        text: Input text.
        max_keywords: Maximum keywords to return.

    Returns:
        List of keywords.
    """
    # Match words: English (2+ chars), Chinese (1+ char)
    words = re.findall(r'\b[a-zA-Z]{2,}\b|[\u4e00-\u9fff]+', text.lower())

    # Remove duplicates while preserving order, filter stop words
    seen = set()
    keywords = []
    for word in words:
        if word not in seen and word not in _STOP_WORDS:
            seen.add(word)
            keywords.append(word)
            if len(keywords) >= max_keywords:
                break

    return keywords
