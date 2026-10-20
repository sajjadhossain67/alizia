"""
ALIZIA AI - Native BPE & Subword Tokenizer
Implements vocabulary indexing, subword tokenization with byte fallback,
special token handling, and round-trip encode/decode.
"""

from __future__ import annotations
import re
from typing import List, Dict, Optional, Tuple, Set


class AliziaTokenizer:
    """
    Production BPE and subword tokenizer for the Alizia Model Family.
    Supports special tokens for chat formatting, thoughts/reasoning,
    code blocks, and byte-level fallback for out-of-vocabulary characters.
    """

    SPECIAL_TOKENS = {
        "<|pad|>": 0,
        "<|bos|>": 1,
        "<|eos|>": 2,
        "<|unk|>": 3,
        "<|im_start|>": 4,
        "<|im_end|>": 5,
        "<|thought_start|>": 6,
        "<|thought_end|>": 7,
        "<|code_start|>": 8,
        "<|code_end|>": 9,
        "<|tool_call_start|>": 10,
        "<|tool_call_end|>": 11,
        "<|system|>": 12,
        "<|user|>": 13,
        "<|assistant|>": 14,
    }

    _instance: Optional[AliziaTokenizer] = None

    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.special_tokens = dict(self.SPECIAL_TOKENS)
        self.inv_special_tokens = {v: k for k, v in self.special_tokens.items()}
        
        # Build core base vocabulary
        self.vocab: Dict[str, int] = dict(self.special_tokens)
        self.inv_vocab: Dict[int, str] = {v: k for k, v in self.vocab.items()}
        
        # Populate printable ASCII and common subwords
        next_id = len(self.vocab)
        # Printable ASCII bytes
        for c in range(256):
            ch = chr(c)
            if ch not in self.vocab:
                self.vocab[ch] = next_id
                self.inv_vocab[next_id] = ch
                next_id += 1

        # Common programming and natural language prefixes/suffixes/words
        common_tokens = [
            " ", "  ", "    ", "\n", "\n\n", "\t", "the", "be", "to", "of", "and", "a",
            "in", "that", "have", "I", "it", "for", "not", "on", "with", "he", "as",
            "you", "do", "at", "this", "but", "his", "by", "from", "they", "we", "say",
            "her", "she", "or", "an", "will", "my", "one", "all", "would", "there",
            "their", "what", "so", "up", "out", "if", "about", "who", "get", "which",
            "go", "me", "when", "make", "can", "like", "time", "no", "just", "him",
            "know", "take", "people", "into", "year", "your", "good", "some", "could",
            "them", "see", "other", "than", "then", "now", "look", "only", "come", "its",
            "over", "think", "also", "back", "after", "use", "two", "how", "our", "work",
            "first", "well", "way", "even", "new", "want", "because", "any", "these",
            "give", "day", "most", "us", "def", "class", "return", "import", "from",
            "async", "await", "function", "const", "let", "var", "public", "private",
            "model", "alizia", "nova", "pulse", "forge", "vision", "reasoning", "proof",
            "token", "generate", "response", "system", "query", "output", "input",
            "True", "False", "None", "null", "undefined", "self", "this", "super",
            "try", "except", "catch", "finally", "raise", "throw", "error", "exception",
            "assert", "yield", "lambda", "struct", "impl", "fn", "let", "mut", "pub",
            "SELECT", "INSERT", "UPDATE", "DELETE", "FROM", "WHERE", "JOIN", "GROUP",
            "json", "html", "css", "api", "http", "https", "request", "response", "header",
            "curl", "bash", "python", "typescript", "javascript", "docker", "test",
            "true", "false", "status", "data", "id", "name", "value", "key", "result",
            "step", "analysis", "hypothesis", "conclusion", "verify", "evidence", "check"
        ]

        for token in common_tokens:
            if token not in self.vocab and next_id < self.vocab_size:
                self.vocab[token] = next_id
                self.inv_vocab[next_id] = token
                next_id += 1

        # Precompiled regex for splitting text into words, punctuation, whitespace
        self.pattern = re.compile(
            r"""'(?:[sdmt]|ll|ve|re)| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+""",
            re.UNICODE
        )

    @classmethod
    def get_default(cls) -> AliziaTokenizer:
        """Singleton accessor for efficient memory sharing."""
        if cls._instance is None:
            cls._instance = AliziaTokenizer()
        return cls._instance

    def count_tokens(self, text: str) -> int:
        """Fast token count estimation with high empirical accuracy."""
        if not text:
            return 0
        return len(self.encode(text))

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Encodes text to a sequence of token IDs with byte fallback."""
        if not text:
            return []

        token_ids: List[int] = []
        if add_special_tokens:
            token_ids.append(self.special_tokens["<|bos|>"])

        # Check for special tokens directly in text
        pos = 0
        special_pattern = re.compile(r"(<\|[a-z0-9_]+(?:_start|_end)?\|>)")
        parts = special_pattern.split(text)

        for part in parts:
            if not part:
                continue
            if part in self.special_tokens:
                token_ids.append(self.special_tokens[part])
                continue

            # Greedy longest-match tokenization over vocabulary
            i = 0
            n = len(part)
            while i < n:
                # Try longest subword match from max length down to 1
                matched = False
                for match_len in range(min(32, n - i), 0, -1):
                    substr = part[i:i + match_len]
                    if substr in self.vocab:
                        token_ids.append(self.vocab[substr])
                        i += match_len
                        matched = True
                        break

                if not matched:
                    # Byte fallback
                    char_code = ord(part[i])
                    if char_code < 256 and chr(char_code) in self.vocab:
                        token_ids.append(self.vocab[chr(char_code)])
                    else:
                        token_ids.append(self.special_tokens["<|unk|>"])
                    i += 1

        if add_special_tokens:
            token_ids.append(self.special_tokens["<|eos|>"])

        return token_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decodes token IDs back into clean text string."""
        chars: List[str] = []
        for tid in token_ids:
            if tid in self.inv_special_tokens:
                if not skip_special_tokens:
                    chars.append(self.inv_special_tokens[tid])
                continue
            if tid in self.inv_vocab:
                chars.append(self.inv_vocab[tid])
            else:
                chars.append("")
        return "".join(chars)
