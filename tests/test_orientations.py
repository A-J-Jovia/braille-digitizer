import sys
from pathlib import Path
import cv2
import numpy as np

# Add api directory to sys.path
api_dir = Path(__file__).resolve().parent.parent / "api"
sys.path.insert(0, str(api_dir))

import braille_engine as be

sample_path = Path(__file__).resolve().parent.parent / "samples" / "sample_braille_sheet.jpg"
img = cv2.imread(str(sample_path))

common_words = {
    'the', 'and', 'for', 'are', 'can', 'trees', 'is', 'year', 'growing', 
    'redwoods', 'sequoias', 'in', 'on', 'at', 'to', 'of', 'a', 'it', 'was', 'as'
}

for rot in [0, 180]:
    cur_img = img if rot == 0 else cv2.rotate(img, cv2.ROTATE_180)
    gray = cv2.cvtColor(cur_img, cv2.COLOR_BGR2GRAY)
    enhanced, shadow = be.preprocess_shadow_map(gray)
    dots = be.detect_dot_centroids(shadow)
    doc, cells = be.decode_page_lattice(dots, gray.shape)
    text = be.translate_document(doc, lang='EN')
    
    words = [w.strip('.,:;!?\'\"-') for w in text.lower().split()]
    matches = sum(1 for w in words if w in common_words or (len(w) >= 4 and w.isalpha()))
    print(f"Rotation {rot}° -> Score: {matches}")
    print("Sample lines:")
    lines = [l for l in text.splitlines() if l.strip()]
    for l in lines[:4]:
        print(f"  {l}")
    print("-" * 40)

