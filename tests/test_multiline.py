import sys
import io
from pathlib import Path

# Add api directory to sys.path
api_dir = Path(__file__).resolve().parent.parent / "api"
sys.path.insert(0, str(api_dir))

import index
import braille_engine as be
from fastapi.testclient import TestClient

client = TestClient(index.app)
multiline_test = "the quick brown fox\njumps over the lazy dog\n123 456 789"

img = be.generate_braille_image(multiline_test)
buf = io.BytesIO()
img.save(buf, format='PNG')
buf.seek(0)

res = client.post('/api/recognize', files={'file': ('multiline.png', buf, 'image/png')})
data = res.json()
print("Decoded English:")
print(data['english_text'])
print("\nDecoded Russian:")
print(data['russian_text'])
print(f"\nDots: {data['dots_count']}, Cells: {data['cells_count']}")
assert data['english_text'] == multiline_test, f"Mismatch:\nExpected:\n{multiline_test}\nGot:\n{data['english_text']}"
print("\nMULTILINE TEST PASSED WITH 100% ACCURACY!")

