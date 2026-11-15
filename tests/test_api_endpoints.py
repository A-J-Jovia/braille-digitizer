import sys
import io
import base64
from pathlib import Path

# Add api directory to sys.path
api_dir = Path(__file__).resolve().parent.parent / "api"
sys.path.insert(0, str(api_dir))

import index
import braille_engine as be
from fastapi.testclient import TestClient

client = TestClient(index.app)

test_phrases = [
    "hello world",
    "braille digitizer",
    "123 abc"
]

print("=== TESTING FASTAPI ENDPOINTS ===")
for phrase in test_phrases:
    print(f"\n[Test] Phrase: '{phrase}'")
    # 1. Generate image using braille_engine
    pil_img = be.generate_braille_image(phrase)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    
    # 2. Call /api/recognize via multipart form
    files = {"file": ("test.png", buf, "image/png")}
    res = client.post("/api/recognize", files=files)
    assert res.status_code == 200, f"Error: {res.status_code} {res.text}"
    data = res.json()
    print("  Status:", data["status"])
    print("  English Decoded:", repr(data["english_text"]))
    print("  Russian Decoded:", repr(data["russian_text"]))
    print("  Dots Count:", data["dots_count"])
    print("  Cells Count:", data["cells_count"])
    print("  Overlay image present:", data["overlay_image"].startswith("data:image/png;base64,"))
    assert data["english_text"] == phrase, f"Mismatch: expected '{phrase}', got '{data['english_text']}'"

print("\nALL API ENDPOINTS TESTED AND PASSED WITH 100% ACCURACY!")

