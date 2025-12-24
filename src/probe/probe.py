import math
from pathlib import Path
import pdfplumber


def sample_pages(pdf_path: str, n: int = 3):
    """Return a list of page indices to sample (0-based).

    Picks start, middle, end when possible, otherwise evenly spaced.
    """
    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
    if total == 0:
        return []
    if n <= 0:
        return []
    if n == 1:
        return [0]
    # choose start, middle, end when n==3
    if n == 3 and total >= 3:
        return [0, total // 2, total - 1]
    # otherwise evenly spaced
    step = max(1, total / (n - 1))
    indices = [min(total - 1, int(round(i * step))) for i in range(n)]
    # ensure unique and sorted
    indices = sorted(set(indices))
    return indices


def analyze_page(pdf_path: str, page_number: int) -> dict:
    """Analyze a single page and return structural signals."""
    with pdfplumber.open(pdf_path) as pdf:
        if page_number < 0 or page_number >= len(pdf.pages):
            raise IndexError("page_number out of range")
        page = pdf.pages[page_number]
        # text
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        text_length = len(text)
        # images
        try:
            images = page.images or []
            image_count = len(images)
        except Exception:
            image_count = 0
        # tables
        try:
            tables = page.find_tables()
            table_count = len(tables) if tables is not None else 0
        except Exception:
            # fallback to heuristic
            table_count = 0
        is_text_layer = text_length > 20  # heuristic
        return {
            "page_number": page_number,
            "text_length": text_length,
            "image_count": image_count,
            "table_count": table_count,
            "is_text_layer": is_text_layer,
        }


def compute_complexity(per_page_signals: list) -> float:
    """Compute a simple complexity score for the document.

    Higher score = more complex (more images/tables/low text coverage).
    """
    if not per_page_signals:
        return 0.0
    scores = []
    for s in per_page_signals:
        score = 0.0
        score += s.get("image_count", 0) * 1.5
        score += s.get("table_count", 0) * 2.0
        # penalize pages with very low text
        if not s.get("is_text_layer", False):
            score += 3.0
        # small boost for long text (reduces complexity)
        text_len = s.get("text_length", 0)
        if text_len > 500:
            score -= 1.0
        scores.append(max(0.0, score))
    # average and normalize
    avg = sum(scores) / len(scores)
    # scale to 0..10 roughly
    normalized = min(10.0, avg / 2.0)
    return round(normalized, 3)


def recommend_action(per_page_signals: list) -> str:
    """Simple recommendation based on signals."""
    if not per_page_signals:
        return "unknown"
    text_layers = sum(1 for s in per_page_signals if s.get("is_text_layer"))
    tables = sum(s.get("table_count", 0) for s in per_page_signals)
    images = sum(s.get("image_count", 0) for s in per_page_signals)
    if text_layers >= len(per_page_signals) * 0.66:
        return "parse_text"
    if tables >= len(per_page_signals):
        return "parse_tables"
    if text_layers == 0 and images > 0:
        return "require_ocr"
    # default
    return "mixed"


def probe_document(pdf_path: str, n_samples: int = 3) -> dict:
    pdf_path = str(pdf_path)
    indices = sample_pages(pdf_path, n_samples)
    per_page = []
    for idx in indices:
        try:
            per_page.append(analyze_page(pdf_path, idx))
        except Exception as e:
            per_page.append({"page_number": idx, "error": str(e)})
    complexity = compute_complexity([p for p in per_page if "error" not in p])
    action = recommend_action([p for p in per_page if "error" not in p])
    summary = {
        "sampled_pages": indices,
        "per_page": per_page,
        "complexity_score": complexity,
        "recommended_action": action,
    }
    return summary
