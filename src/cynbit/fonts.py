"""Register bundled DejaVu fonts with ReportLab so PDFs render symbols like ≥ → ≈ correctly."""
from __future__ import annotations

from functools import lru_cache

from .config import PROJECT_ROOT

FONT_DIR = PROJECT_ROOT / "assets" / "fonts"


@lru_cache(maxsize=1)
def register_fonts() -> tuple[str, str, str]:
    """Return (regular, bold, mono) font names; falls back to built-in Helvetica/Courier."""
    try:
        from reportlab.lib.fonts import addMapping
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont

        files = {"DejaVu": "DejaVuSans.ttf", "DejaVu-Bold": "DejaVuSans-Bold.ttf",
                 "DejaVu-Oblique": "DejaVuSans-Oblique.ttf", "DejaVu-BoldOblique": "DejaVuSans-BoldOblique.ttf",
                 "DejaVuMono": "DejaVuSansMono.ttf"}
        for name, fn in files.items():
            pdfmetrics.registerFont(TTFont(name, str(FONT_DIR / fn)))
        addMapping("DejaVu", 0, 0, "DejaVu")
        addMapping("DejaVu", 1, 0, "DejaVu-Bold")
        addMapping("DejaVu", 0, 1, "DejaVu-Oblique")
        addMapping("DejaVu", 1, 1, "DejaVu-BoldOblique")
        return "DejaVu", "DejaVu-Bold", "DejaVuMono"
    except Exception:  # fonts missing -> still produce a PDF
        return "Helvetica", "Helvetica-Bold", "Courier"
