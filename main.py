import sys
from pathlib import Path

# Ensure api directory is on sys.path
api_dir = Path(__file__).resolve().parent / "api"
if str(api_dir) not in sys.path:
    sys.path.insert(0, str(api_dir))

from api.index import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
