"""
Generate QR codes for the slides from slides/_build/links.json, then rebuild the deck.
Fill links.json first, e.g.
  {"colab": "https://colab.research.google.com/github/<you>/vietbrew-demo/blob/main/notebooks/01_VietBrew_Live_Demo.ipynb",
   "dashboard": "https://<your-app>.streamlit.app", "github": "https://github.com/<you>/vietbrew-demo"}
Run:  python src/make_qr.py   then   cd slides/_build && node build_demo_class.js
"""
import json
from pathlib import Path

import qrcode

B = Path(__file__).resolve().parents[1] / "slides" / "_build"
links = json.loads((B / "links.json").read_text(encoding="utf-8"))
for key, url in links.items():
    out = B / "img" / f"qr_{key}.png"
    if url:
        qrcode.make(url, box_size=12, border=2).save(out)
        print("QR", key, "->", out)
    elif out.exists():
        out.unlink()
