"""Unit tests for keyword extraction."""

import pytest
from memory.utils.keyword_extractor import extract_keywords


class TestKeywordExtractor:
    """Test suite for keyword extraction."""

    def test_extract_keywords_basic(self):
        """Test basic keyword extraction from English text."""
        text = "Python and FastAPI are great for building web APIs"
        keywords = extract_keywords(text)
        assert len(keywords) > 0
        assert any("python" in kw.lower() for kw in keywords)
        assert any("fastapi" in kw.lower() for kw in keywords)

    def test_extract_keywords_empty(self):
        """Test extraction from empty string."""
        keywords = extract_keywords("")
        assert keywords == []

    def test_extract_keywords_none(self):
        """Test extraction from None."""
        keywords = extract_keywords(None)
        assert keywords == []

    def test_extract_keywords_stop_words_filtered(self):
        """Test that stop words are filtered out."""
        text = "the quick brown fox jumps over the lazy dog"
        keywords = extract_keywords(text)
        # "the" and "over" should not be in keywords
        assert not any(kw.lower() == "the" for kw in keywords)
        assert not any(kw.lower() == "over" for kw in keywords)

    def test_extract_keywords_max_limit(self):
        """Test that max_keywords parameter is respected."""
        text = "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10"
        keywords = extract_keywords(text, max_keywords=5)
        assert len(keywords) <= 5

    def test_extract_keywords_chinese(self):
        """Test extraction from Chinese text."""
        text = "Python和FastAPI是构建网络API的好工具"
        keywords = extract_keywords(text)
        assert len(keywords) > 0

    def test_extract_keywords_mixed_language(self):
        """Test extraction from mixed English and Chinese."""
        text = "Python是一个很好的编程语言，FastAPI用于构建API"
        keywords = extract_keywords(text)
        assert len(keywords) > 0

    def test_extract_keywords_short_words_filtered(self):
        """Test that words shorter than 2 chars are filtered."""
        text = "I am a developer"
        keywords = extract_keywords(text)
        # Single char words should be filtered
        assert not any(len(kw) < 2 for kw in keywords)

    def test_extract_keywords_duplicates_removed(self):
        """Test that duplicate keywords are removed."""
        text = "test test test example example"
        keywords = extract_keywords(text)
        # Should not have duplicates
        assert len(keywords) == len(set(keywords))

    def test_extract_keywords_case_insensitive(self):
        """Test that extraction is case-insensitive."""
        text1 = "Python FastAPI"
        text2 = "python fastapi"
        keywords1 = extract_keywords(text1)
        keywords2 = extract_keywords(text2)
        # Should extract same keywords regardless of case
        assert len(keywords1) == len(keywords2)
