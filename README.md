# Braille Digitizer — Serverless Web App (Vercel Ready)

A production-ready, lightweight Optical Braille Recognition (OBR) and Synthesis web application engineered for deployment on **Vercel Serverless Functions**.

---

## 🚀 Key Highlights & Architecture

- **Weight-Free Whole-Page Grid Lattice Reconstruction**:
  - Reconstructs canonical 2×3 Braille cells directly from whole-page dot centroids without cropping or losing empty-dot spatial frames.
  - Enforces physical 2-column cell width limits ($\le 1.25 \times d_x$) to prevent multi-letter clustering.
  - Uses $k$-d tree / vectorized spatial queries for dynamic dot pitch estimation ($d_x, d_y$).
  - Fast CPU runtime (<40ms per page) with **zero GPU/CUDA requirements** and **zero gigabyte weight checkpoints**.
- **Serverless & Lightweight**:
  - Complies strictly with Vercel's 50MB (zipped) / 250MB (uncompressed) serverless limit.
  - Dependencies: `fastapi`, `uvicorn`, `python-multipart`, `numpy`, `pillow`, `opencv-python-headless`.
- **Bilingual & Grade-1 UEB Translation**:
  - Unified English Braille (UEB) Grade 1 alphabet & contractions.
  - Number mode support with `#` (`3456`) numerical sign handling.
  - Russian Braille alphabet.
  - Robust word spacing using inter-cell gap thresholding ($\ge 3.2 \times d_x$).
- **Dual-Capability Interactive UI**:
  - **Recognize Braille**: Drag-and-drop image upload, instant visual lattice overlay, bilingual transcription, and 1-click sample buttons.
  - **Generate Braille**: Type text to synthesize standard raised-dot Braille charts with 1-click test in recognizer or PNG download.

---

## 📁 Project Structure

```
vercel_app/
├── api/
│   ├── index.py              # FastAPI serverless endpoints (/api/recognize, /api/generate, /api/health)
│   ├── braille_engine.py      # Core lattice reconstruction & UEB translation engine
│   └── requirements.txt      # Dependency specification for serverless build
├── public/
│   └── index.html            # Tailwind CSS responsive frontend
├── requirements.txt          # Root dependency specification
├── vercel.json               # Modern Vercel URL rewrite routing configuration
├── .gitignore
└── README.md
```

---

## 💻 Local Testing

You can run and test the app locally with Uvicorn:

```bash
cd vercel_app
pip install -r requirements.txt
uvicorn api.index:app --reload --port 8000
```

Open `http://localhost:8000` or open `public/index.html` in your browser.

---

## 🌐 Deploying to Vercel

### Method 1: Deploy with Vercel CLI (Fastest)

1. Install the Vercel CLI globally (if not already installed):
   ```bash
   npm install -g vercel
   ```

2. Navigate to the `vercel_app` directory:
   ```bash
   cd C:\Users\ajjov\.gemini\antigravity\scratch\Braille_Digitizer\vercel_app
   ```

3. Run the deploy command:
   ```bash
   vercel
   ```
   Follow the prompts:
   - *Set up and deploy?* **Y**
   - *Which scope?* Select your account
   - *Link to existing project?* **N**
   - *Project name?* `braille-digitizer` (or your preferred name)
   - *Directory located?* `./`
   - *Want to modify settings?* **N**

4. To deploy to production:
   ```bash
   vercel --prod
   ```

---

### Method 2: Deploy via GitHub (Recommended for Continuous Deployment)

1. Initialize a git repository inside `vercel_app`:
   ```bash
   cd C:\Users\ajjov\.gemini\antigravity\scratch\Braille_Digitizer\vercel_app
   git init
   git add .
   git commit -m "Initial commit of Braille Digitizer Vercel web app"
   ```

2. Create a new repository on [GitHub](https://github.com/new) (e.g., `braille-digitizer`).

3. Push your code:
   ```bash
   git remote add origin https://github.com/<your-username>/braille-digitizer.git
   git branch -M main
   git push -u origin main
   ```

4. Go to [Vercel Dashboard](https://vercel.com/new):
   - Click **Add New...** -> **Project**.
   - Import your `braille-digitizer` GitHub repository.
   - Leave the Framework Preset as **Other** (Vercel automatically detects `vercel.json`).
   - Click **Deploy**.

Your application will be live globally on your Vercel URL (e.g. `https://braille-digitizer.vercel.app`)!

