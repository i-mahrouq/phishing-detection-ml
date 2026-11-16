"""Shared inference-safe keyword features."""
import re

KEYWORDS = ('urgent', 'verify', 'verification', 'password', 'account', 'login',
            'signin', 'security', 'suspend', 'invoice', 'payment', 'bank',
            'confirm', 'credential', 'wallet', 'winner', 'click')
WORDS = re.compile(r'\b(?:' + '|'.join(KEYWORDS) + r')\b', re.I)
FRAGMENTS = re.compile('|'.join(KEYWORDS), re.I)


def suspicious_count(text, whole_words=True):
    return sum(1 for _ in (WORDS if whole_words else FRAGMENTS).finditer(text))
