"""
Braille Engine: Core Optical Braille Recognition (OBR) and Translation Library
Lightweight, pure CV/SciPy implementation designed for local notebooks and Vercel serverless deployment.
"""

import numpy as np
from PIL import Image, ImageDraw
import cv2

try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ==============================================================================
# 1. UNIFIED ENGLISH BRAILLE (UEB), RUSSIAN & CONTRACTION DICTIONARIES
# ==============================================================================

# Standard UEB Grade 1 Alphabet & Symbols
UEB_ENGLISH_MAP = {
    '1': 'a', '12': 'b', '14': 'c', '145': 'd', '15': 'e',
    '124': 'f', '1245': 'g', '125': 'h', '24': 'i', '245': 'j',
    '13': 'k', '123': 'l', '134': 'm', '1345': 'n', '135': 'o',
    '1234': 'p', '12345': 'q', '1235': 'r', '234': 's', '2345': 't',
    '136': 'u', '1236': 'v', '2456': 'w', '1346': 'x', '13456': 'y', '1356': 'z',
    '2': ',', '23': ';', '25': ':', '256': '.', '235': '!', '236': '?',
    '36': '-', '3': "'", '356': '"', '26': '?'
}

# Common UEB Contractions per Project Report Section 6.1
UEB_CONTRACTIONS = {
    '12346': 'and', '123456': 'for', '12356': 'of', '2346': 'the', '23456': 'with',
    '16': 'ch', '126': 'gh', '146': 'sh', '1456': 'th', '156': 'wh',
    '1246': 'ed', '12456': 'er', '1256': 'ou', '246': 'ow', '34': 'st',
    '346': 'ing', '345': 'ar', '35': 'in', '26': 'en'
}

# Number prefix and digit mapping
NUM_SIGN = '3456'
BRAILLE_DIGITS = {
    '1': '1', '12': '2', '14': '3', '145': '4', '15': '5',
    '124': '6', '1245': '7', '125': '8', '24': '9', '245': '0'
}

# Russian Braille Alphabet
RUSSIAN_MAP = {
    '1': 'а', '12': 'б', '2456': 'в', '1245': 'г', '145': 'д', '15': 'е',
    '16': 'ё', '245': 'ж', '1356': 'з', '24': 'и', '12346': 'й', '13': 'к',
    '123': 'л', '134': 'м', '1345': 'н', '135': 'о', '1234': 'п', '1235': 'р',
    '234': 'с', '2345': 'т', '136': 'у', '124': 'ф', '125': 'х', '14': 'ц',
    '12345': 'ч', '156': 'ш', '1346': 'щ', '12356': 'ъ', '2346': 'ы', '23456': 'ь',
    '246': 'э', '1256': 'ю', '1246': 'я',
    '2': ',', '23': ';', '25': ':', '256': '.', '235': '!', '236': '?',
    '36': '-', '3': "'"
}

CHAR_TO_DOTS = {v: k for k, v in UEB_ENGLISH_MAP.items() if len(v) == 1}
DIGIT_TO_DOTS = {v: k for k, v in BRAILLE_DIGITS.items()}

def bits_to_dot_string(bits):
    """Converts a 6-element binary tuple/list to a dot number string (e.g. [1, 1, 0, 0, 0, 0] -> '12')."""
    return "".join(str(i + 1) for i, b in enumerate(bits) if b)

def pattern_id_to_bits(pattern_id):
    """Converts a numeric pattern ID (0-63) to a 6-bit array."""
    return [(pattern_id >> i) & 1 for i in range(6)]

def bits_to_pattern_id(bits):
    """Converts a 6-bit array to a pattern ID (0-63)."""
    return sum(int(b) * (1 << i) for i, b in enumerate(bits))

def translate_dots(dot_str, lang="EN", in_number_mode=False):
    """Translates a dot sequence string into a symbol."""
    if not dot_str:
        return "", in_number_mode

    if dot_str == NUM_SIGN:
        return "", True

    if in_number_mode:
        if dot_str in BRAILLE_DIGITS:
            return BRAILLE_DIGITS[dot_str], True
        else:
            in_number_mode = False

    if lang == "EN":
        if dot_str in UEB_ENGLISH_MAP:
            return UEB_ENGLISH_MAP[dot_str], in_number_mode
        if dot_str in UEB_CONTRACTIONS:
            return UEB_CONTRACTIONS[dot_str], in_number_mode
    else:
        if dot_str in RUSSIAN_MAP:
            return RUSSIAN_MAP[dot_str], in_number_mode

    # Unknown pattern fallback per project report Section 5.4.6
    return f"[dots {dot_str}]", in_number_mode

# ==============================================================================
# 2. SHADOW-ASSISTED PREPROCESSING & DOT DETECTION
# ==============================================================================

def preprocess_shadow_map(gray_img):
    """
    Applies CLAHE (8x8 tiles, clipLimit 2.0) and Morphological Blackhat
    to isolate shadows cast by raised Braille dots under side lighting.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray_img)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    shadow_map = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, kernel)
    return enhanced, shadow_map

def detect_dot_centroids(shadow_map, min_area=3, max_area=140, min_dim=2, max_dim=20):
    """
    Detects candidate dot centroids using Otsu binarization and connected components.
    """
    blurred = cv2.GaussianBlur(shadow_map, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    valid_dots = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        if (min_dim <= w <= max_dim) and (min_dim <= h <= max_dim) and (min_area <= area <= max_area):
            valid_dots.append((float(centroids[i][0]), float(centroids[i][1])))

    return np.array(valid_dots)

# ==============================================================================
# 3. K-D TREE PITCH ESTIMATION & CANONICAL LATTICE RECONSTRUCTION
# ==============================================================================

def estimate_lattice_parameters(dots_xy):
    """
    Estimates horizontal dot pitch (dx), vertical dot pitch (dy), character pitch,
    and word spacing threshold using nearest-neighbor spatial queries via cKDTree
    (or vectorized numpy fallback if scipy is omitted).
    """
    if len(dots_xy) < 4:
        return 12.0, 12.0, 38.4

    dots_arr = np.array(dots_xy, dtype=np.float32)
    dy_candidates = []

    if HAS_SCIPY:
        tree = cKDTree(dots_arr)
        for x, y in dots_arr:
            indices = tree.query_ball_point([x, y], r=35)
            for j in indices:
                x2, y2 = dots_arr[j]
                dx = abs(x2 - x)
                dy = y2 - y
                if 6 <= dy <= 30 and dx <= 6:
                    dy_candidates.append(dy)
    else:
        for x, y in dots_arr:
            dxs = np.abs(dots_arr[:, 0] - x)
            dys = dots_arr[:, 1] - y
            mask = (dys >= 6) & (dys <= 30) & (dxs <= 6)
            dy_candidates.extend(dys[mask].tolist())

    median_dy = float(np.median(dy_candidates)) if dy_candidates else 12.0
    # Standard Braille intra-cell aspect ratio is 1:1
    median_dx = median_dy
    word_thresh = median_dx * 3.2

    return median_dx, median_dy, word_thresh

def decode_page_lattice(dots_xy, image_shape):
    """
    Robust whole-page lattice decoder:
      1. Groups dot centroids into reading lines.
      2. Enforces physical 2-column cell width limit (<= 1.25 * dx), preventing cell merging.
      3. Anchors canonical 3-row levels relative to line baseline, preserving empty-dot positions.
      4. Inserts true word space tokens when inter-cell gap exceeds word threshold.
      5. Reconstructs 6-bit binary Braille cells in true reading order.
    """
    if len(dots_xy) == 0:
        return [], []

    median_dx, median_dy, word_thresh = estimate_lattice_parameters(dots_xy)
    H, W = image_shape[:2]

    # 1. Line grouping: group dots belonging to the same 3-row Braille line
    dots_sorted_y = dots_xy[np.argsort(dots_xy[:, 1])]
    lines = []
    for dot in dots_sorted_y:
        placed = False
        for line in lines:
            line_ys = [d[1] for d in line]
            min_y = min(line_ys)
            if (min_y - 0.4 * median_dy) <= dot[1] <= (min_y + 2.4 * median_dy):
                line.append(dot)
                placed = True
                break
        if not placed:
            lines.append([dot])

    # Sort lines from top to bottom
    lines.sort(key=lambda l: np.mean([d[1] for d in l]))

    decoded_document = []
    all_decoded_cells = []

    for line_dots in lines:
        arr = np.array(line_dots)
        arr = arr[np.argsort(arr[:, 0])]  # Left-to-right sorting
        min_y = np.min(arr[:, 1])

        # 2. Cell clustering enforcing physical cell width limit
        cells = []
        curr_cell_dots = [arr[0]]

        for k in range(1, len(arr)):
            d_curr = arr[k]
            proposed_min_x = min(min(d[0] for d in curr_cell_dots), d_curr[0])
            proposed_max_x = max(max(d[0] for d in curr_cell_dots), d_curr[0])
            proposed_width = proposed_max_x - proposed_min_x

            # A physical Braille cell has width <= 1.25 * median_dx
            if proposed_width <= (median_dx * 1.25):
                curr_cell_dots.append(d_curr)
            else:
                dx_gap = d_curr[0] - max(d[0] for d in curr_cell_dots)
                cells.append((curr_cell_dots, dx_gap))
                curr_cell_dots = [d_curr]
        cells.append((curr_cell_dots, 0))

        # 3. Decode canonical 2x3 grid positions
        line_tokens = []
        for c_dots, next_gap in cells:
            c_arr = np.array(c_dots)
            c_min_x = np.min(c_arr[:, 0])
            c_max_x = np.max(c_arr[:, 0])
            has_two_cols = (c_max_x - c_min_x) >= (median_dx * 0.5)
            mid_x = (c_min_x + c_max_x) / 2.0 if has_two_cols else c_min_x

            bits = [0] * 6
            for pt in c_dots:
                # Column 0 (left: dots 1,2,3) or Column 1 (right: dots 4,5,6)
                col = 0 if (not has_two_cols or pt[0] <= mid_x) else 1
                # Row 0 (top), Row 1 (middle), Row 2 (bottom)
                row = int(round((pt[1] - min_y) / max(median_dy, 1.0)))
                row = min(2, max(0, row))
                bit_idx = row if col == 0 else row + 3
                bits[bit_idx] = 1

            pattern_id = bits_to_pattern_id(bits)
            dot_str = bits_to_dot_string(bits)

            cell_box = (
                int(c_min_x - 3),
                int(min_y - 3),
                int(max(c_max_x - c_min_x + 6, median_dx * 1.2)),
                int(median_dy * 2.4)
            )

            cell_info = {
                "bbox": cell_box,
                "bits": bits,
                "dot_str": dot_str,
                "pattern_id": pattern_id,
                "center": ((c_min_x + c_max_x) / 2.0, (min_y + median_dy))
            }
            line_tokens.append(cell_info)
            all_decoded_cells.append(cell_info)

            # Insert word space token if gap exceeds word_thresh
            if next_gap >= word_thresh:
                space_info = {
                    "bbox": None,
                    "bits": [0] * 6,
                    "dot_str": "",
                    "pattern_id": 0,
                    "is_space": True,
                    "center": (c_max_x + next_gap / 2.0, min_y + median_dy)
                }
                line_tokens.append(space_info)

        decoded_document.append(line_tokens)

    return decoded_document, all_decoded_cells

# ==============================================================================
# 4. TRANSLATION TO FORMATTED TEXT
# ==============================================================================

def translate_document(decoded_document, lang="EN"):
    """Translates decoded document tokens into formatted multiline text."""
    output_lines = []
    for line in decoded_document:
        line_chars = []
        in_number_mode = False
        for token in line:
            if token.get("is_space"):
                line_chars.append(" ")
                in_number_mode = False
                continue
            symbol, in_number_mode = translate_dots(token["dot_str"], lang=lang, in_number_mode=in_number_mode)
            line_chars.append(symbol)
        output_lines.append("".join(line_chars).strip())
    return "\n".join(output_lines)

# ==============================================================================
# 5. VISUAL LATTICE OVERLAY RENDERING
# ==============================================================================

def render_lattice_overlay(base_img, decoded_cells, detected_dots=None):
    """
    Renders visual feedback:
      - Detected dot centroids as green circles.
      - Canonical 2x3 reconstructed cell bounding boxes in blue.
      - Pattern ID and dot strings annotated in yellow.
    """
    if len(base_img.shape) == 2:
        vis = cv2.cvtColor(base_img, cv2.COLOR_GRAY2RGB)
    else:
        vis = base_img.copy()

    # Draw dot centroids
    if detected_dots is not None and len(detected_dots) > 0:
        for pt in detected_dots:
            cv2.circle(vis, (int(pt[0]), int(pt[1])), 2, (0, 255, 0), -1)

    # Draw cell bounding boxes and annotations
    for cell in decoded_cells:
        if cell.get("is_space") or not cell.get("bbox"):
            continue
        x, y, w, h = cell["bbox"]
        cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 100, 0), 1)
        tag = cell["dot_str"] if cell["dot_str"] else "empty"
        cv2.putText(vis, tag, (x, max(10, y - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

    return vis

# ==============================================================================
# 6. TEXT-TO-BRAILLE GENERATOR (Report Section 5.4.8)
# ==============================================================================

def generate_braille_image(text, dot_radius=3, dot_spacing=12, char_spacing=28, line_spacing=45):
    """
    Synthesizes a clean Braille document/chart image from typed English text.
    Fulfills project report Section 5.4.8 (Text-to-Braille Generator).
    Supports English alphabet, punctuation, and Braille numbers with '#' number sign.
    """
    lines = text.splitlines() if text else [""]

    parsed_lines = []
    max_cells_in_line = 1
    for line in lines:
        cell_seq = []
        in_num_mode = False
        for ch in line.lower():
            if ch == " ":
                cell_seq.append(None)
                in_num_mode = False
            elif ch in DIGIT_TO_DOTS:
                if not in_num_mode:
                    cell_seq.append(NUM_SIGN)
                    in_num_mode = True
                cell_seq.append(DIGIT_TO_DOTS[ch])
            elif ch in CHAR_TO_DOTS:
                cell_seq.append(CHAR_TO_DOTS[ch])
                in_num_mode = False
        parsed_lines.append(cell_seq)
        if len(cell_seq) > max_cells_in_line:
            max_cells_in_line = len(cell_seq)

    width = max(260, (max_cells_in_line + 2) * char_spacing)
    height = max(160, (len(lines) + 2) * line_spacing)

    img = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(img)

    for row_idx, cell_seq in enumerate(parsed_lines):
        y_top = (row_idx + 1) * line_spacing
        col_x = char_spacing

        for cell in cell_seq:
            if cell is None:
                col_x += char_spacing
                continue

            for d in cell:
                d_idx = int(d)
                col = 0 if d_idx in (1, 2, 3) else 1
                row = (d_idx - 1) % 3
                dx = col_x + col * dot_spacing
                dy = y_top + row * dot_spacing
                draw.ellipse([dx - dot_radius, dy - dot_radius, dx + dot_radius, dy + dot_radius], fill=0)

            col_x += char_spacing

    return img
