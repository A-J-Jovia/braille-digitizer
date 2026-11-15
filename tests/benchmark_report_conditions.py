"""
Comprehensive 6-Condition Synthetic Benchmark Suite
Fulfills Project Report Chapter 6 (Sections 6.1 and 6.2).

Tests 30 charts per condition x 46 Braille symbols per chart = 1,380 cells per condition across:
  1. Clean charts
  2. Side-light brightness gradient (100% -> 55% across page)
  3. Gaussian sensor noise (sigma = 18)
  4. JPEG compression (Quality = 35)
  5. Gaussian blur (sigma = 2 px)
  6. Combined (gradient + blur + noise)

Random dot pitch (18 to 29 px), dot radius (5 to 8 px), cell and row spacing.
Reports:
  - Cell accuracy (matches ground truth symbol)
  - Number of charts decoded perfectly (46/46 correct)
  - Average CPU processing time per chart
"""

import sys
import time
import random
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import braille_engine as be

# 46 Symbols in seven columns per Report Section 6.1
SYMBOLS_46 = [
    ('A', [1]), ('B', [1, 2]), ('C', [1, 4]), ('D', [1, 4, 5]), ('E', [1, 5]),
    ('F', [1, 2, 4]), ('G', [1, 2, 4, 5]), ('H', [1, 2, 5]), ('I', [2, 4]), ('J', [2, 4, 5]),
    ('K', [1, 3]), ('L', [1, 2, 3]), ('M', [1, 3, 4]), ('N', [1, 3, 4, 5]), ('O', [1, 3, 5]),
    ('P', [1, 2, 3, 4]), ('Q', [1, 2, 3, 4, 5]), ('R', [1, 2, 3, 5]), ('S', [2, 3, 4]), ('T', [2, 3, 4, 5]),
    ('U', [1, 3, 6]), ('V', [1, 2, 3, 6]), ('W', [2, 4, 5, 6]), ('X', [1, 3, 4, 6]), ('Y', [1, 3, 4, 5, 6]),
    ('Z', [1, 3, 5, 6]), ('#', [3, 4, 5, 6]), ('and', [1, 2, 3, 4, 6]),
    ('for', [1, 2, 3, 4, 5, 6]), ('of', [1, 2, 3, 5, 6]), ('the', [2, 3, 4, 6]), ('with', [2, 3, 4, 5, 6]),
    ('ch', [1, 6]), ('gh', [1, 2, 6]), ('sh', [1, 4, 6]), ('th', [1, 4, 5, 6]), ('wh', [1, 5, 6]),
    ('ed', [1, 2, 4, 6]), ('er', [1, 2, 4, 5, 6]), ('ou', [1, 2, 5, 6]), ('ow', [2, 4, 6]),
    ('st', [3, 4]), ('ing', [3, 4, 6]), ('ar', [3, 4, 5]), ('in', [3, 5]), ('en', [2, 6])
]

DOT_OFFSETS = {
    1: (0, 0), 2: (0, 1), 3: (0, 2),
    4: (1, 0), 5: (1, 1), 6: (1, 2)
}

def render_reference_chart(pitch, radius, seed=None):
    if seed is not None:
        random.seed(seed)
    cols = 7
    char_spacing = int(pitch * random.uniform(2.7, 3.1))
    row_spacing = int(pitch * random.uniform(3.7, 4.1))
    margin = 40
    W = margin * 2 + cols * char_spacing
    H = margin * 2 + 7 * row_spacing
    img = Image.new('L', (W, H), color=255)
    draw = ImageDraw.Draw(img)

    gt_cells = []
    for idx, (sym, dots) in enumerate(SYMBOLS_46):
        r_idx = idx // cols
        c_idx = idx % cols
        cell_x = margin + c_idx * char_spacing
        cell_y = margin + r_idx * row_spacing
        gt_pattern = ''.join(str(d) for d in sorted(dots))
        gt_cells.append((sym, gt_pattern))
        for d in dots:
            col_off, row_off = DOT_OFFSETS[d]
            cx = cell_x + col_off * pitch
            cy = cell_y + row_off * pitch
            draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=35)

    return np.array(img), gt_cells

# Degradation pipelines
def apply_clean(img):
    return img

def apply_gradient(img):
    H, W = img.shape
    grad = np.linspace(1.0, 0.55, W, dtype=np.float32)
    return np.clip(img.astype(np.float32) * grad[np.newaxis, :], 0, 255).astype(np.uint8)

def apply_sensor_noise(img, sigma=18):
    noise = np.random.normal(0, sigma, img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

def apply_jpeg(img, quality=35):
    _, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)

def apply_blur(img, sigma=2.0):
    ksize = int(2 * round(3 * sigma) + 1)
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)

def apply_combined(img):
    out = apply_gradient(img)
    out = apply_blur(out, sigma=1.8)
    out = apply_sensor_noise(out, sigma=16)
    return out

def run_benchmark(charts_per_condition=30):
    print("=" * 80)
    print("BRAILLE DIGITIZER — 6-CONDITION SYNTHETIC BENCHMARK EVALUATION")
    print(f"Charts per condition: {charts_per_condition} | Cells per chart: 46 | Total: {charts_per_condition * 46} cells/condition")
    print("=" * 80)

    conditions = [
        ("Clean Charts", apply_clean),
        ("Side-Light Gradient (100%->55%)", apply_gradient),
        ("Gaussian Sensor Noise (sigma=18)", apply_sensor_noise),
        ("JPEG Compression (Q=35)", apply_jpeg),
        ("Gaussian Blur (sigma=2px)", apply_blur),
        ("Combined (Gradient+Blur+Noise)", apply_combined)
    ]

    results = []

    for name, deg_fn in conditions:
        print(f"\nEvaluating Condition: {name}...")
        total_cells = charts_per_condition * len(SYMBOLS_46)
        correct_cells = 0
        perfect_charts = 0
        total_time_ms = 0.0

        for seed in range(charts_per_condition):
            pitch = random.randint(18, 29)
            radius = random.randint(5, 8)
            clean_gray, gt_cells = render_reference_chart(pitch, radius, seed=seed * 100 + 7)
            
            # Apply degradation
            test_img = deg_fn(clean_gray)
            
            # Measure CPU execution time
            t0 = time.perf_counter()
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enh = clahe.apply(test_img)
            blurred = cv2.GaussianBlur(enh, (5, 5), 0)
            binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 15)
            kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel5)
            num, _, stats, centroids = cv2.connectedComponentsWithStats(opened, connectivity=8)
            areas = [stats[k, cv2.CC_STAT_AREA] for k in range(1, num)]
            A_ref = np.percentile(areas, 85) if areas else 10
            dots = [centroids[k] for k in range(1, num) if 0.35 * A_ref <= stats[k, cv2.CC_STAT_AREA] <= 2.2 * A_ref]
            
            doc, cells = be.decode_page_lattice(np.array(dots, dtype=np.float32), test_img.shape)
            t1 = time.perf_counter()
            total_time_ms += (t1 - t0) * 1000.0

            # Evaluate cell match
            chart_correct = 0
            for c_idx, cell in enumerate(cells):
                if c_idx < len(gt_cells):
                    if cell["dot_str"] == gt_cells[c_idx][1]:
                        chart_correct += 1

            correct_cells += chart_correct
            if chart_correct == len(gt_cells):
                perfect_charts += 1

        accuracy_pct = (correct_cells / total_cells) * 100.0
        avg_time = total_time_ms / charts_per_condition
        results.append({
            "condition": name,
            "total_cells": total_cells,
            "correct_cells": correct_cells,
            "accuracy": accuracy_pct,
            "perfect_charts": perfect_charts,
            "avg_time_ms": avg_time
        })
        print(f"  -> Accuracy: {correct_cells}/{total_cells} ({accuracy_pct:.2f}%) | Perfect Charts: {perfect_charts}/{charts_per_condition} | Mean CPU Time: {avg_time:.2f} ms")

    # Format Markdown Table matching Report Table 6.2
    print("\n" + "=" * 80)
    print("FINAL BENCHMARK SUMMARY (Matches Project Report Table 6.2)")
    print("=" * 80)
    print(f"{'Condition':<35} | {'Cells':<7} | {'Correct':<7} | {'Accuracy':<9} | {'Perfect':<8} | {'CPU Time (ms)':<13}")
    print("-" * 90)
    for r in results:
        print(f"{r['condition']:<35} | {r['total_cells']:<7} | {r['correct_cells']:<7} | {r['accuracy']:>6.2f}%   | {r['perfect_charts']:>2}/{charts_per_condition:<4} | {r['avg_time_ms']:>8.2f} ms")
    print("=" * 80)

if __name__ == "__main__":
    run_benchmark(charts_per_condition=30)
