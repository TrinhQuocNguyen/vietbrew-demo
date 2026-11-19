# ☕ VietBrew — From Data to Decision with Python & AI

**upGrad × Golden Gate University · MBA Bilingual Demo Class · Advanced Data Analytics with Python** (03/12/2026)

> VietBrew is a **fictional** Vietnamese coffee brand. All data in this repository are **simulated**, modelled on the
> Vietnamese market (Lunar New Year seasonality, double-day sales, cash on delivery, the 2025 province merger).
> *VietBrew là doanh nghiệp hư cấu; toàn bộ dữ liệu là dữ liệu mô phỏng.*

**The case:** *“We have VND 2 billion for Tết 2027 marketing. Where should it go?”*

## Open the notebooks

| Notebook | What it does |
|---|---|
| `notebooks/00_Hello_VietBrew.ipynb` | 2-minute check that Google Colab works on your device |
| `notebooks/01_VietBrew_Live_Demo.ipynb` | The full demo: Parts 1–6 (Python, data traps, Tết forecast, K-Means, AI, decision memo) |

In Google Colab: **File → Open notebook → GitHub**, paste this repository's URL, choose a notebook, then **Runtime → Run all**.

## Data (`data/`)

| File | Rows | Content |
|---|---|---|
| `raw/vietbrew_sales_2023_2026.csv.gz` | 1,501,436 | One row per product line of each order, 01/01/2023 – 30/11/2026, as exported (messy on purpose) |
| `products.csv` | 12 | SKU, Vietnamese/English name, category, list prices, unit cost |
| `channel_costs.csv` | 4 | Illustrative fee and logistics assumptions per channel |
| `marketing_spend.csv` | 188 | Ad spend per month and channel (VND) |
| `reviews.csv` | 3,000 | Vietnamese customer reviews (with teencode, missing accents, emoji) |
| `tet_dates.csv` | 5 | Tết (Lunar New Year) dates 2023–2027 |
| `province_mapping_2025.csv` | 63 | Old province → new provincial unit (34 units from 01/07/2025) → region |
| `instructor/` | — | Reference labels for the reviews and the simulated customer segments (for checking AI and K-Means) |

Raw sales columns: `order_id, order_datetime, customer_id, channel, province, sku, product_name, category, quantity,
unit_price` (Vietnamese format, e.g. `145.000`), `discount_vnd, payment_method, order_status`.

## Dashboard

`dashboard/app.py` — Streamlit app (Overview · Tết 2027 planner · Channels · Customers · Reviews).
Run locally: `streamlit run dashboard/app.py`. Deploy free on Streamlit Community Cloud with main file `dashboard/app.py`.

## Rebuild everything (instructor)

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt     # Windows; use .venv/bin on macOS/Linux
python src/generate_data.py        # synthetic data (seed 2026, ~1 minute)
python src/build_outputs.py        # clean parquet, dashboard aggregates, slides/_build/slide_data.json
python src/build_notebooks.py --github <user>/<repo> --dashboard https://<app>.streamlit.app
cd slides/_build && npm install && node build_demo_class.js    # slide deck (.pptx)
```
