# Shared preparation: create stable content hashes for duplicate detection.

import hashlib


# Identity stage: hash text to identify matching content.
def digest(text):
    return hashlib.sha256(text.encode('utf-8', errors='replace')).hexdigest()
