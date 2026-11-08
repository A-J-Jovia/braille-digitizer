import io
import base64
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent))
try:
    import braille_engine as be
except ImportError:
    from api import braille_engine as be

try:
    from html_page import HTML_CONTENT
except ImportError:
    from api.html_page import HTML_CONTENT

app = FastAPI(title="Braille Digitizer OBR API", description="Serverless Braille Recognition and Synthesis API")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return HTMLResponse(content=HTML_CONTENT)

class GenerateRequest(BaseModel):
    text: str

@app.get("/api/health")
@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "Braille Digitizer OBR"}

@app.post("/api/recognize")
@app.post("/recognize")
async def recognize_braille(file: UploadFile = File(...), rotation: int = Form(0)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_np = np.array(image)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        # 1. Whole-page lattice grid decoding with auto-orientation
        doc, all_cells, dots, proc_gray, applied_rot = be.decode_page_with_auto_orient(
            gray, force_rotate=rotation, auto_orient=(rotation == 0)
        )

        # 2. Bilingual translation
        en_text = be.translate_document(doc, lang="EN")
        ru_text = be.translate_document(doc, lang="RU")

        # 3. Render lattice visual overlay
        vis = be.render_lattice_overlay(proc_gray, all_cells, detected_dots=dots)
        is_success, buffer = cv2.imencode(".png", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        if is_success:
            overlay_b64 = "data:image/png;base64," + base64.b64encode(buffer).decode("utf-8")
        else:
            overlay_b64 = None

        return {
            "status": "success",
            "english_text": en_text,
            "russian_text": ru_text,
            "dots_count": len(dots),
            "cells_count": len(all_cells),
            "overlay_image": overlay_b64,
            "rotation_applied": applied_rot
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate")
@app.post("/generate")
async def generate_braille(req: GenerateRequest):
    try:
        if not req.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty.")

        chart_img = be.generate_braille_image(req.text)
        buf = io.BytesIO()
        chart_img.save(buf, format="PNG")
        b64_chart = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

        return {
            "status": "success",
            "chart_image": b64_chart
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
