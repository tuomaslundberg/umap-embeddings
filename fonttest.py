import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties
from PIL import ImageFont

def find_font_supporting(char, candidates=None):
    if candidates is None:
        candidates = [
            "Noto Naskh Arabic", "Noto Sans Arabic", "Scheherazade", "Amiri",
            "Droid Arabic Naskh", "Droid Sans Fallback", "DejaVu Sans"
        ]
    for name in candidates:
        try:
            path = fm.findfont(name, fallback_to_default=False)
        except Exception:
            continue
        try:
            ImageFont.truetype(path, size=16).getmask(char)
            return path
        except Exception:
            continue
    return None

# example usage: pick a font that supports an Urdu character (ARABIC LETTER YEH BARREE U+06EE)
arabic_test_char = '\u06CC'  # or use '\u06ee' etc.
arabic_font_path = find_font_supporting(arabic_test_char)
if arabic_font_path:
    arabic_fp = FontProperties(fname=arabic_font_path)
    # use arabic_fp when annotating Urdu/Arabic/Chinese text
else:
    arabic_fp = None
print(f"Selected font path: {arabic_font_path}")