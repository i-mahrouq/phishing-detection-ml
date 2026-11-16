"""Stateless lexical URL features. No DNS, HTTP, webpage features, or fitting."""
from collections import Counter
import ipaddress
import math
from urllib.parse import urlsplit
import numpy as np
import pandas as pd
from src.preprocessing.urls import normalize_url
from .common import suspicious_count

URL_COLUMNS = ('url_length', 'hostname_length', 'dot_count', 'subdomain_count',
               'digit_count', 'hyphen_count', 'special_character_count', 'is_https',
               'hostname_is_ip', 'url_entropy', 'suspicious_keyword_count',
               'path_length', 'query_parameter_count')


class URLFeatures:
    """Extract URL structure rather than fetching content or resolving hosts."""

    def transform(self, values) -> pd.DataFrame:
        """Return finite float32 lexical features in the immutable URL_COLUMNS order.

        Original punctuation, case and path lengths remain model inputs. The
        subdomain count is a label-count heuristic, not public-suffix extraction.
        """
        rows = []
        for value in values:
            record = normalize_url(value)  # Reuse validated offline URL parsing.
            raw = record['url_original']
            parts = urlsplit(raw.strip())
            host = parts.hostname.rstrip('.')
            try:
                ipaddress.ip_address(host)
                is_ip = 1
            except ValueError:
                is_ip = 0
            n = len(raw)
            entropy = -sum((count/n) * math.log2(count/n) for count in Counter(raw).values())
            rows.append((n, len(host), raw.count('.'), 0 if is_ip else max(0, len(host.split('.'))-2),
                         sum(c.isdigit() for c in raw), raw.count('-'), sum(not c.isalnum() for c in raw),
                         int(parts.scheme.lower() == 'https'), is_ip, entropy,
                         suspicious_count(raw, whole_words=False), len(parts.path),
                         sum(bool(p) for p in parts.query.split('&')) if parts.query else 0))
        result = pd.DataFrame(rows, columns=URL_COLUMNS, dtype=np.float32)
        if not np.isfinite(result.to_numpy()).all():
            raise ValueError('Nonfinite URL features')
        return result

    def schema(self):
        return dict(columns=list(URL_COLUMNS), dtype='float32', fitted=False,
                    input='url_original (preserves case, percent escapes, query order and punctuation)',
                    subdomain_definition='max(hostname label count - 2, 0), zero for IPs; heuristic, not public-suffix aware',
                    scaling='none; raw finite lexical features',
                    forbidden_features='labels, source identifiers, webpage-derived columns')
