import os
from pathlib import Path

def find_default_pdf():
    candidates = [
        'battle_viewer/Card_ID List_EN.pdf',
        'battle_viewer/Card_ID List_JP.pdf',
        'Card_ID List_EN.pdf',
        'Card_ID List_JP.pdf'
    ]
    for c in candidates:
        if os.path.exists(c):
            return Path(c)
    return None
