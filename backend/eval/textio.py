"""Reading text files that may have been saved by a Windows editor.

Notepad and several other Windows editors write UTF-8 with a byte order
mark. Decoding that as plain utf-8 leaves a stray "﻿" on the first
line, which turns into a confusing JSON parse error or an unmatched first
question. Every hand-edited file in eval/ is read through here.
"""

import codecs
from pathlib import Path


def read_text_utf8(path: str | Path) -> str:
    """Read a file as UTF-8, tolerating a leading byte order mark."""
    data = Path(path).read_bytes()
    if data.startswith(codecs.BOM_UTF8):
        return data.decode("utf-8-sig")
    return data.decode("utf-8")
