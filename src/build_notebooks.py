"""
Builds the two Colab notebooks from code (so they stay reproducible):
  notebooks/00_Hello_VietBrew.ipynb        pre-class check (2 minutes)
  notebooks/01_VietBrew_Live_Demo.ipynb    the live demo, Parts 1-6
Run:  python src/build_notebooks.py
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]

# Optional:  python src/build_notebooks.py --github <user>/<repo> --dashboard https://<app>.streamlit.app
import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--github", default="<YOUR-GITHUB>/vietbrew-demo")
_ap.add_argument("--dashboard", default="https://<your-app>.streamlit.app")
ARGS = _ap.parse_args()
NB = ROOT / "notebooks"
NB.mkdir(exist_ok=True)


def md(s):
    return nbf.v4.new_markdown_cell(s.strip("\n"))


def code(s):
    return nbf.v4.new_code_cell(s.strip("\n"))


SETUP = r'''
# ▶ SETUP — run this cell first · chạy ô này trước tiên
import os, re, time, unicodedata, urllib.request
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
pio.renderers.default = "colab" if IN_COLAB else "plotly_mimetype+notebook_connected"
pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 160)

# Where is the data? Local folder (VS Code) or online (Colab)
GITHUB_DATA = "https://raw.githubusercontent.com/__GITHUB__/main/data/"   # ← giảng viên sửa dòng này
DATA = next((p for p in ["../data/", "data/", "vietbrew-demo/data/"] if os.path.exists(p + "products.csv")), GITHUB_DATA)


def get(path):
    """Return a local path; download once if the data lives online."""
    if not DATA.startswith("http"):
        return DATA + path
    local = path.replace("/", "__")
    if not os.path.exists(local):
        urllib.request.urlretrieve(DATA + path, local)
    return local


print("Data source:", DATA, "| Colab:", IN_COLAB)
'''

SETUP = SETUP.replace("__GITHUB__", ARGS.github)

hello = nbf.v4.new_notebook()
hello.cells = [
    md("""
# ☕ Hello VietBrew — pre-class check
**upGrad × Golden Gate University · MBA Bilingual Demo Class · Advanced Data Analytics with Python**

Notebook 2 phút để kiểm tra máy của anh/chị đã sẵn sàng cho buổi học.
1. Menu **Runtime → Run all** (hoặc bấm ▶ ở từng ô).
2. Nếu thấy biểu đồ ở cuối trang: anh/chị đã sẵn sàng. Không cần cài đặt gì thêm.

> VietBrew là doanh nghiệp **hư cấu**; toàn bộ số liệu là **dữ liệu mô phỏng** theo đặc điểm thị trường Việt Nam.
"""),
    code(SETUP),
    code('''
products = pd.read_csv(get("products.csv"))
products[["sku", "product_name", "category", "list_price_2026"]]
'''),
    code('''
tet = pd.read_csv(get("tet_dates.csv"))
tet["weekday"] = pd.to_datetime(tet["tet_date"]).dt.day_name()
tet          # Tết moves every year — remember this table, it matters in class
'''),
    code('''
fig = px.bar(products.sort_values("list_price_2026"), x="list_price_2026", y="product_name_en", orientation="h",
             color="category", title="VietBrew product range (VND, 2026 list price)", height=420)
fig.show()
print("✅ Ready for class. Hẹn gặp anh/chị tối 03/12/2026, 20:00.")
'''),
]

demo = nbf.v4.new_notebook()
C = demo.cells
C += [
    md("""
# ☕ VietBrew: From Data to Decision with Python & AI
**upGrad × Golden Gate University — MBA Bilingual Demo Class · Advanced Data Analytics with Python**

> **CEO question · Câu hỏi của CEO:** *“We have **VND 2 billion** for Tết 2027 marketing. Where should it go?”*
> Tết 2027 is on **6 February 2027**.

VietBrew is a **fictional** coffee brand (cà phê hạt, drip bag, cold brew, hộp quà Tết) selling on **Shopee, TikTok Shop, Lazada** and its **own website**.
All numbers are **simulated data** modelled on the Vietnamese market.

| Part | Question | Tool |
|---|---|---|
| 1 · Why Python | Can we even open 4 years of sales? | pandas |
| 2 · Basic analysis | What happened? (and 3 Vietnam-specific data traps) | pandas, Plotly |
| 3 · Advanced analysis | When will Tết demand peak? Who are our customers? | Tết alignment, K-Means |
| 4 · AI as analyst | Can AI do it — and where does it go wrong? | Gemini in Colab |
| 5 · Dashboard | How do managers see it every day? | Streamlit |
| 6 · Insight → Decision | So what? Now what? | Decision memo |

**How to use · Cách dùng:** run cells from top to bottom (**Shift + Enter**). Cells marked **🔧 TRY IT** invite you to change one value and run again.
"""),
    code(SETUP),
    md("""
---
## Part 1 · Why Python? — Vì sao dùng Python?
Excel stops at **1,048,576 rows** per sheet. VietBrew's 4-year sales export is one row per product line of each order.
"""),
    code('''
t0 = time.time()
raw = pd.read_csv(get("raw/vietbrew_sales_2023_2026.csv.gz"), dtype={"unit_price": str})
print(f"Read {len(raw):,} rows × {raw.shape[1]} columns in {time.time() - t0:.1f} seconds")
print(f"Excel's limit is 1,048,576 rows → this file is {len(raw) / 1_048_576:.2f}× too big for one sheet")
'''),
    code('''
raw.sample(5, random_state=7)      # 5 random rows: what does the data look like?
'''),
    md("""
---
## Part 2 · Basic analysis — and three Vietnam-specific data traps
Before any chart, clean the data. Ba “cái bẫy” rất Việt Nam:
1. **Giá kiểu Việt Nam** `145.000` — máy tính hiểu dấu chấm là dấu thập phân.
2. **Tỉnh thành** viết đủ kiểu, và **sáp nhập 63 → 34 tỉnh từ 01/07/2025**.
3. **Dòng trùng** do hệ thống xuất dữ liệu.

### Trap 1 · `145.000` is not 145
"""),
    code('''
naive_price = pd.to_numeric(raw["unit_price"], errors="coerce")       # the "quick fix" many people try
print(raw["unit_price"].head(3).tolist(), "→", naive_price.head(3).tolist())
print(f"Rows that could not be read at all: {naive_price.isna().sum():,}  (e.g. '1.250.000')")
print(f"Revenue with the quick fix: {(raw['quantity'] * naive_price).sum() / 1e9:,.2f} billion VND")

raw["unit_price"] = raw["unit_price"].str.replace(".", "", regex=False).astype(int)   # the correct fix
print(f"Revenue after the correct fix: {(raw['quantity'] * raw['unit_price']).sum() / 1e9:,.1f} billion VND")
# Tip: pd.read_csv(..., thousands=".") does the same thing while reading the file
'''),
    md("### Trap 2 · One city, many spellings — and a map that changed on 1 July 2025"),
    code('''
print("Different spellings in the 'province' column:", raw["province"].nunique())
raw.loc[raw["province"].str.contains("Hồ Chí Minh|HCM|Sài Gòn|Ho Chi Minh|hcm", regex=True), "province"].value_counts()
'''),
    code('''
# A naive chart: orders typed as "Bình Dương" — did VietBrew lose this market in mid-2025?
bd = raw[raw["province"].str.contains("Bình Dương|Binh Duong|bình dương", regex=True)]
bd_month = bd.groupby(bd["order_datetime"].str[:7])["order_id"].nunique()
fig = px.bar(bd_month.loc["2024-07":], title="Orders typed as 'Bình Dương' per month — panic?", labels={"value": "orders", "order_datetime": "month"})
fig.add_vline(x="2025-07", line_dash="dash", line_color="red")
fig.update_layout(showlegend=False, height=350)
fig.show()
'''),
    md("""
No panic: from **1/7/2025** Bình Dương and Bà Rịa – Vũng Tàu are part of **TP. Hồ Chí Minh** (Nghị quyết 202/2025/QH15).
Fix: normalise the text (bỏ dấu, chữ thường, bỏ “TP.”/“Tỉnh”) and map **old → new** with a 63-row table.
"""),
    code('''
mapping = pd.read_csv(get("province_mapping_2025.csv"))          # 63 old provinces → 34 new units
ALIASES = {"hcm": "ho chi minh", "tphcm": "ho chi minh", "sai gon": "ho chi minh", "ho chi minh city": "ho chi minh",
           "hn": "ha noi", "hanoi": "ha noi", "tt hue": "thua thien hue", "vung tau": "ba ria - vung tau",
           "br-vt": "ba ria - vung tau", "ba ria vung tau": "ba ria - vung tau"}


def norm(s):
    s = str(s).replace("đ", "d").replace("Đ", "D")
    s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn").lower().strip()
    s = re.sub(r"^(tp\\.?\\s*|thanh pho\\s+|tinh\\s+)", "", s).strip()
    return ALIASES.get(s, s)


lookup = {norm(o): n for o, n in zip(mapping["old_province"], mapping["new_province"])}
lookup.update({norm(n): n for n in mapping["new_province"]})
spellings = raw["province"].unique()
fix = {sp: lookup.get(norm(sp), "UNKNOWN") for sp in spellings}
raw["province_clean"] = raw["province"].map(fix)
raw["region"] = raw["province_clean"].map(dict(zip(mapping["new_province"], mapping["region"])))
print(f"{len(spellings)} spellings → {raw['province_clean'].nunique()} provinces · unmatched rows: {(raw['province_clean'] == 'UNKNOWN').sum()}")
'''),
    md("### Trap 3 · Duplicated rows from the export"),
    code('''
print("Duplicated rows:", raw.duplicated().sum())
sales = raw.drop_duplicates().copy()

# build the analysis table
sales["date"] = pd.to_datetime(sales["order_datetime"]).dt.normalize()
sales["gross_vnd"] = sales["quantity"] * sales["unit_price"]
sales["sales_vnd"] = sales["gross_vnd"] - sales["discount_vnd"]          # what the customer pays
sales["is_cod"] = sales["payment_method"].eq("COD")
tet = pd.read_csv(get("tet_dates.csv"))
tet_dates = pd.to_datetime(tet["tet_date"]).sort_values().to_numpy()
pos = np.searchsorted(tet_dates, sales["date"].to_numpy())
sales["days_to_tet"] = (tet_dates[pos] - sales["date"].to_numpy()).astype("timedelta64[D]").astype(int)
sales["tet_season"] = pd.DatetimeIndex(tet_dates[pos]).year
valid = sales[sales["order_status"] != "Cancelled"]
print(f"Clean table: {len(sales):,} rows · {sales['order_id'].nunique():,} orders · {sales['customer_id'].nunique():,} customers")
print(f"Revenue 2023–11/2026 (excl. cancelled): {valid['sales_vnd'].sum() / 1e9:,.1f} billion VND")
'''),
    md("""
### What happened? — Doanh thu theo tháng
> ### ❓ POLL
> **February 2026 revenue was about 5.5× February 2025. What happened?**
> A. The TikTok campaign worked · B. A new product · C. A price increase · D. Something else
"""),
    code('''
monthly = valid.groupby(valid["date"].dt.to_period("M"))["sales_vnd"].sum() / 1e9
monthly.index = monthly.index.astype(str)
fig = px.bar(monthly, title="VietBrew revenue per month (billion VND)", labels={"value": "billion VND", "date": ""})
fig.update_layout(showlegend=False, height=380)
fig.show()
print(f"Feb 2026 vs Feb 2025: {monthly['2026-02'] / monthly['2025-02']:.1f}×   |   Jan 2026 vs Jan 2025: {monthly['2026-01'] / monthly['2025-01'] - 1:+.0%}")
'''),
    code('''
tet.assign(weekday=pd.to_datetime(tet["tet_date"]).dt.day_name())     # 💡 the answer is in this table
'''),
    md("""
**Answer: D.** Tết moved from **29/1/2025** to **17/2/2026**. Comparing calendar months year-on-year is misleading in Vietnam.
Compare **Tết seasons** instead (60 days before each Tết):
"""),
    code('''
season = valid[valid["days_to_tet"].between(0, 60)].groupby("tet_season")["sales_vnd"].sum() / 1e9
print(season.round(1).to_string())
print(f"Season 2026 vs season 2025: {season[2026] / season[2025] - 1:+.0%}")
'''),
    md("### Who sells? — Kênh bán hàng"),
    code('''
mix = valid.pivot_table(index=valid["date"].dt.year, columns="channel", values="sales_vnd", aggfunc="sum")
mix = (mix.div(mix.sum(axis=1), axis=0) * 100).round(1)
fig = px.bar(mix, title="Channel share of revenue (%)", labels={"value": "% of revenue", "date": "year"}, text_auto=True)
fig.update_layout(height=380)
fig.show()
mix
'''),
    md("#### 🔧 TRY IT — change the province and run again"),
    code('''
PROVINCE = "Đà Nẵng"      # try: "Hà Nội", "TP. Hồ Chí Minh", "Cần Thơ", "Khánh Hòa"
p = valid[valid["province_clean"] == PROVINCE]
m = p.groupby([p["date"].dt.to_period("M").astype(str), "channel"])["sales_vnd"].sum().div(1e6).reset_index()
px.bar(m, x="date", y="sales_vnd", color="channel", title=f"{PROVINCE}: revenue per month (million VND)", height=360).show()
'''),
    md("""
---
## Part 3 · Advanced analysis
### 3.1 The Tết calendar trap — Lịch âm “đánh lừa” mô hình quốc tế
Most global templates use the **Gregorian** calendar. Let us plot gift-box sales on calendar dates, then on **days before Tết**.
"""),
    code('''
gift = valid[valid["category"] == "Tết Gift Box"]
daily_gift = gift.groupby(["tet_season", "date", "days_to_tet"], as_index=False)["quantity"].sum()
cal = daily_gift[daily_gift["tet_season"].isin([2024, 2025, 2026])].copy()
cal["day_of_season"] = (cal["date"] - pd.to_datetime((cal["tet_season"] - 1).astype(str) + "-12-01")).dt.days
cal = cal[cal["day_of_season"].between(0, 89)]
fig = px.line(cal, x="day_of_season", y="quantity", color="tet_season",
              title="Gift boxes sold per day, calendar view (day 0 = 1 December)", labels={"quantity": "boxes / day", "day_of_season": "days since 1 Dec"})
fig.update_layout(height=360)
fig.show()
'''),
    code('''
al = daily_gift[daily_gift["tet_season"].isin([2024, 2025, 2026]) & (daily_gift["days_to_tet"] <= 60)]
fig = px.line(al, x="days_to_tet", y="quantity", color="tet_season",
              title="Same data, aligned on Tết: the three seasons overlap", labels={"quantity": "boxes / day", "days_to_tet": "days before Tết"})
fig.update_xaxes(autorange="reversed")
fig.update_layout(height=360)
fig.show()
peak = al.loc[al.groupby("tet_season")["quantity"].idxmax(), ["tet_season", "date", "days_to_tet"]]
peak
'''),
    md("### 3.2 Forecast Tết 2027 (6/2/2027) — Dự báo mùa quà Tết"),
    code('''
TET_2027 = pd.Timestamp("2027-02-06")
full = [2024, 2025, 2026]                                     # 2023 season is cut by the data start (01/01/2023)
seasons = al.groupby("tet_season")["quantity"].sum()
growth = float(np.exp(np.log(seasons / seasons.shift(1)).dropna().mean()))
shape = (al.groupby(["tet_season", "days_to_tet"])["quantity"].sum()
           .groupby(level=0).transform(lambda x: x / x.sum()).groupby(level=1).mean())
shape = shape.reindex(range(61), fill_value=0); shape /= shape.sum()
total_2027 = seasons[2026] * growth
fc = pd.DataFrame({"days_to_tet": shape.index, "units": shape.values * total_2027})
fc["date"] = TET_2027 - pd.to_timedelta(fc["days_to_tet"], unit="D")
fc = fc.sort_values("date"); fc["cum"] = fc["units"].cumsum() / total_2027
print(f"Season totals: {seasons.to_dict()} · average growth {growth - 1:+.0%}")
print(f"Forecast Tết 2027: ~{total_2027:,.0f} gift boxes")
print(f"10% sold by {fc.loc[fc['cum'] >= .1, 'date'].iloc[0]:%d/%m/%Y} · peak {fc.loc[fc['units'].idxmax(), 'date']:%d/%m/%Y} · 80% by {fc.loc[fc['cum'] >= .8, 'date'].iloc[0]:%d/%m/%Y}")

# the naive way: copy last season onto the same calendar dates
prev = al[al["tet_season"] == 2026]
late = prev.loc[prev["date"] + pd.DateOffset(years=1) > TET_2027 - pd.Timedelta(days=3), "quantity"].sum() / prev["quantity"].sum()
print(f"Naive calendar copy: {late:.0%} of demand would land after the last shipping day (03/02/2027)")
wk = fc.groupby(fc["date"].dt.to_period("W-SUN").dt.start_time)["units"].sum().round(-1)
fig = px.bar(wk, title="Forecast gift boxes per week, Tết 2027", labels={"value": "boxes", "date": "week starting"})
fig.add_vline(x=pd.Timestamp("2027-02-03"), line_dash="dash", line_color="red")
fig.update_layout(showlegend=False, height=340)
fig.show()
'''),
    md("""
### 3.3 Who are our customers? — Phân khúc khách hàng (RFM + K-Means)
**RFM** = **R**ecency (mua gần nhất cách đây bao lâu), **F**requency (mua bao nhiêu lần), **M**onetary (chi bao nhiêu).
We add two behaviour signals — share of spend on gift boxes, average discount — and let **K-Means** group similar customers.
"""),
    code('''
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

REF = pd.Timestamp("2026-11-30")
d = valid[valid["order_status"] == "Delivered"]
f = d.groupby("customer_id").agg(last=("date", "max"), frequency=("order_id", "nunique"), monetary=("sales_vnd", "sum"))
f["recency"] = (REF - f["last"]).dt.days
g = valid.assign(gift=valid["category"].eq("Tết Gift Box") * valid["sales_vnd"]).groupby("customer_id")[["gift", "sales_vnd", "discount_vnd", "gross_vnd"]].sum()
f["gift_share"] = g["gift"] / g["sales_vnd"]
f["discount_rate"] = g["discount_vnd"] / g["gross_vnd"]
f = f.drop(columns="last").dropna()

X = f[["recency", "frequency", "monetary", "gift_share", "discount_rate"]].copy()
X[["frequency", "monetary"]] = np.log1p(X[["frequency", "monetary"]])
f["cluster"] = KMeans(n_clusters=4, n_init=10, random_state=42).fit_predict(StandardScaler().fit_transform(X))
f.groupby("cluster")[["recency", "frequency", "monetary", "gift_share", "discount_rate"]].mean().round(2)
'''),
    code('''
# Name the clusters with simple business rules — a human decision, not the algorithm's
prof = f.groupby("cluster")[["frequency", "gift_share", "discount_rate"]].mean()
names = {}
for col, name, pick in [("frequency", "Loyal Brewers", "idxmax"), ("gift_share", "Tết Gifters", "idxmax"),
                        ("discount_rate", "Deal Hunters", "idxmax"), ("frequency", "One-time Triers", "idxmin")]:
    names[getattr(prof.drop(index=list(names))[col], pick)()] = name
f["segment"] = f["cluster"].map(names)
seg = f.groupby("segment").agg(customers=("recency", "size"), orders=("frequency", "mean"), spend_vnd=("monetary", "mean"),
                               gift_share=("gift_share", "mean"), discount=("discount_rate", "mean"), revenue=("monetary", "sum"))
seg["% customers"] = (seg["customers"] / seg["customers"].sum() * 100).round(1)
seg["% revenue"] = (seg["revenue"] / seg["revenue"].sum() * 100).round(1)
seg.drop(columns="revenue").round(2).sort_values("% revenue", ascending=False)
'''),
    code('''
sample = f.sample(6000, random_state=1)
fig = px.scatter(sample, x="frequency", y="monetary", color="segment", log_x=True, log_y=True, opacity=0.5,
                 title="Each dot is a customer (sample of 6,000)", labels={"monetary": "total spend (VND)", "frequency": "orders"})
fig.update_layout(height=420)
fig.show()
top = f["monetary"].sort_values(ascending=False)
print(f"Top 20% of customers bring {top.head(int(len(top) * .2)).sum() / top.sum():.0%} of revenue")
'''),
    md("""
---
## Part 4 · AI as your analyst — AI làm trợ lý phân tích
### 4.1 Ask in plain Vietnamese (Gemini in Colab)
In Colab, click **Gemini** (✦ icon, top right) or the **Data Science Agent**, then paste one prompt. AI writes and runs the Python for you.

| # | Prompt (copy · dán) |
|---|---|
| 1 | *Dùng DataFrame `valid`, vẽ biểu đồ doanh thu theo kênh bán hàng cho từng năm, đơn vị tỷ đồng.* |
| 2 | *Tỉnh nào có tỷ lệ giao hàng thất bại cao nhất trong mùa Tết 2026 (days_to_tet từ 3 đến 28)? Chỉ xét tỉnh có trên 500 đơn.* |
| 3 | *Which payment method has the highest share of failed deliveries? Show a table and a bar chart.* |
| 4 | *Giải thích ngắn gọn cho CEO (không dùng thuật ngữ kỹ thuật) biểu đồ vừa vẽ cho thấy điều gì.* |

> ⚠️ AI may answer differently each time — that is normal. Always **check the code and the numbers** it produces.

### 4.2 AI reads 3,000 Vietnamese reviews — có cả teencode
"""),
    code('''
reviews = pd.read_csv(get("reviews.csv"))
print(len(reviews), "reviews")
reviews.sample(8, random_state=3)[["channel", "region", "rating", "review_text"]]
'''),
    md("**Step 1 — the old way: keyword rules.** Fast, cheap, but brittle with teencode (`k`, `dc`, `ship lâu quá trời`, no accents…)."),
    code('''
TEEN = [(r"\\b(k|ko|hong|hok)\\b", "khong"), (r"\\b(dc|dk)\\b", "duoc"), (r"\\b(cf|cafe)\\b", "ca phe"), (r"\\b(r|rui)\\b", "roi")]
RULES = [("Service", ["sai san pham", "giao hop", "thieu hang", "tu van", "tra loi", "doi hang"]),
         ("Packaging", ["mop", "dong goi", "bao bi", "rach", "bi be", "thung", "chong soc", "goi hang", "trau xuoc", "bien dang"]),
         ("Delivery", ["giao", "ship", "tre", "cham", "kho trung chuyen", "dung hen", "nhan duoc", "co hang"]),
         ("Price", ["gia", "sale", "hoi dat", "giam sau"]),
         ("Taste", ["vi ", "thom", "ngon", "chua", "khet", "nhat", "gu", "dam", "uong"])]
NEG = ["cham", "lau", "tre", "mop", "be", "rach", "that vong", "sai", "thieu", "nhat", "khet", "gat", "hoi dat", "uot",
       "bien dang", "trau xuoc", "so sai", "khong ai", "khong thay", "khong giong", "khong hop", "xem lai", "cho dai", "bi chua"]
POS = ["nhanh", "dep", "ngon", "thom", "ok", "chu dao", "nhiet tinh", "qua hoi", "5 sao", "dang tien", "hop ly", "chac chan",
       "can than", "mua lai", "dung hen", "than thien", "thich", "dung gu", "ung ho", "dam da"]


def rule_classify(text):
    t = norm(text)
    t = re.sub(r"(\\w)\\1{2,}", r"\\1", t)
    for a, b in TEEN:
        t = re.sub(a, b, t)
    aspect = next((a for a, kws in RULES if any(k in t for k in kws)), "Other")
    neg, pos = sum(k in t for k in NEG), sum(k in t for k in POS)
    return aspect, "Negative" if neg > pos else "Positive" if pos > neg else "Neutral"


reviews[["aspect", "sentiment"]] = reviews["review_text"].apply(lambda t: pd.Series(rule_classify(t)))
reviews[["review_text", "aspect", "sentiment"]].sample(6, random_state=5)
'''),
    md("**Step 2 — ask an LLM (runs in Colab, free, no API key).** `google.colab.ai` lets notebook code call Gemini directly."),
    code('''
N_AI = 100          # 🔧 TRY IT: more reviews = slower; 100 takes about a minute
PROMPT = """You are a customer-experience analyst for a Vietnamese coffee brand.
Each review may contain slang, teencode or missing accents. For EACH review return ONE line:
<review_id>|<aspect>|<sentiment>
aspect: Delivery, Packaging, Taste, Price, Service or Other. sentiment: Positive, Negative or Neutral.
Return only the lines.

Reviews:
"""


def ai_classify(df, batch=25):
    from google.colab import ai
    rows = []
    for i in range(0, len(df), batch):
        chunk = df.iloc[i:i + batch]
        text = "\\n".join(f"{r.review_id}: {r.review_text}" for r in chunk.itertuples())
        answer = ai.generate_text(PROMPT + text)
        rows += [m.groups() for m in re.finditer(r"(RV\\d{5})\\s*\\|\\s*(\\w+)\\s*\\|\\s*(\\w+)", answer)]
    return pd.DataFrame(rows, columns=["review_id", "ai_aspect", "ai_sentiment"])


sample_ai = reviews.sample(N_AI, random_state=11)
if IN_COLAB:
    t0 = time.time()
    ai_out = ai_classify(sample_ai)
    print(f"AI labelled {len(ai_out)} reviews in {time.time() - t0:.0f} s")
    sample_ai = sample_ai.merge(ai_out, on="review_id", how="left")
    display(sample_ai[["review_text", "aspect", "ai_aspect", "sentiment", "ai_sentiment"]].head(10))
else:
    print("Not running in Colab → skipping the live AI call (open this notebook in Colab to run it).")
'''),
    code('''
# Instructor check: compare with the reference labels of the simulated dataset (if available)
try:
    ref = pd.read_csv(get("instructor/reviews_labels.csv"))
    chk = reviews.merge(ref, on="review_id", suffixes=("", "_ref"))
    print(f"Keyword rules — aspect correct: {(chk.aspect == chk.aspect_ref).mean():.0%}, sentiment correct: {(chk.sentiment == chk.sentiment_ref).mean():.0%}")
    if "ai_aspect" in sample_ai:
        c2 = sample_ai.merge(ref, on="review_id", suffixes=("", "_ref"))
        print(f"LLM ({len(c2)} reviews) — aspect correct: {(c2.ai_aspect == c2.aspect_ref).mean():.0%}, sentiment correct: {(c2.ai_sentiment == c2.sentiment_ref).mean():.0%}")
except Exception as e:
    print("Reference labels not available:", e)
'''),
    code('''
neg = reviews[reviews["sentiment"] == "Negative"]
tab = pd.crosstab(neg["aspect"], neg["region"]).reindex(columns=["North", "Central", "South"])
fig = px.bar(tab, barmode="group", title="Negative reviews by topic and region", labels={"value": "reviews"})
fig.update_layout(height=360)
fig.show()
cust_central = sales.drop_duplicates("customer_id")["region"].eq("Central").mean()
print(f"Central region: {cust_central:.0%} of customers but {tab.loc['Delivery', 'Central'] / tab.loc['Delivery'].sum():.0%} of delivery complaints")
'''),
    md("""
### 4.3 The trap — AI answers the question you ask
> **Prompt:** *“Which channel gives the best return on our advertising spend? Recommend how to split next season's budget.”*

A typical AI answer computes **ROAS = GMV ÷ ad spend** (what the platforms report) and recommends the top channel. Here is that calculation:
"""),
    code('''
spend = pd.read_csv(get("marketing_spend.csv"))
last12 = valid[(valid["date"] >= "2025-12-01") & (valid["date"] <= "2026-11-30")]
ads = spend[spend["month"].between("2025-12", "2026-11")].groupby("channel")["ad_spend_vnd"].sum()
roas = (last12.groupby("channel")["sales_vnd"].sum() / ads).sort_values(ascending=False).round(1)
print("Gross ROAS (GMV ÷ ad spend), last 12 months:"); print(roas.to_string())
print(f"\\n🤖 'AI recommendation': put most of the budget on {roas.index[0]}.")
'''),
    md("""
**Would you sign this?** — Anh/chị có ký duyệt không?
GMV counts orders that are later **returned**, **refused at the door (COD)**, sold at a **deep discount**, and paid to **creators/platforms**. Ask the right question: *how much do we keep per 1 VND of ads?*

`contribution = delivered revenue − platform & creator fees − cost of goods − logistics` (assumptions in `channel_costs.csv`)
"""),
    code('''
products = pd.read_csv(get("products.csv"))
costs = pd.read_csv(get("channel_costs.csv")).set_index("channel")
x = last12.merge(products[["sku", "unit_cost"]], on="sku")
x = x.join(costs, on="channel")
delivered, returned = x["order_status"].eq("Delivered"), x["order_status"].eq("Returned")
x["net_revenue"] = np.where(delivered, x["sales_vnd"], 0)
x["fees"] = x["net_revenue"] * (x["platform_commission"] + x["affiliate_commission"] + np.where(x["is_cod"], 0, x["payment_fee_prepaid"]))
x["cogs"] = x["unit_cost"] * x["quantity"] * np.where(delivered, 1, np.where(returned, x["returned_goods_write_off"], 0))
orders = x.drop_duplicates("order_id")
econ = x.groupby("channel")[["sales_vnd", "net_revenue", "fees", "cogs"]].sum()
econ["orders"] = orders.groupby("channel").size()
econ["failed_or_returned"] = orders[orders["order_status"].isin(["Failed delivery", "Returned"])].groupby("channel").size()
econ["logistics"] = econ["orders"] * costs["fulfilment_cost_per_order"] + econ["failed_or_returned"] * costs["reverse_logistics_per_failed_or_returned"]
econ["contribution"] = econ["net_revenue"] - econ["fees"] - econ["cogs"] - econ["logistics"]
econ["ad_spend"] = ads
result = pd.DataFrame({
    "gross ROAS": econ["sales_vnd"] / econ["ad_spend"],
    "kept per 1 VND of ads": econ["contribution"] / econ["ad_spend"],
    "failed+returned %": econ["failed_or_returned"] / econ["orders"] * 100,
    "profit after ads (bn VND)": (econ["contribution"] - econ["ad_spend"]) / 1e9,
}).round(2).sort_values("gross ROAS", ascending=False)
result
'''),
    code('''
fig = go.Figure([go.Bar(name="Gross ROAS (what platforms report)", x=result.index, y=result["gross ROAS"]),
                 go.Bar(name="VND kept per 1 VND of ads", x=result.index, y=result["kept per 1 VND of ads"])])
fig.update_layout(barmode="group", title="The ranking flips when you look at what the business keeps", height=380)
fig.show()
'''),
    md("""
---
## Part 5 · Dashboard — managers do not open notebooks
The same analysis as a live web app (Streamlit). Open it on your phone: filter by channel, region, Tết season.
"""),
    code('''
DASHBOARD_URL = "__DASHBOARD__"       # ← giảng viên điền sau khi deploy (Streamlit Community Cloud)
print("Open:", DASHBOARD_URL)
'''),
    md("""
**How it was built:** one prompt to an AI coding agent in **VS Code** (Claude Code), then human review:
> *“Build a Streamlit dashboard from the parquet files in dashboard/data: KPI tiles (revenue, orders, failed+returned %), monthly revenue by channel, Tết-aligned gift-box curves with the 2027 forecast, gross ROAS vs net return by channel, customer segments, and negative reviews by topic and region. Sidebar filters: date range, channel, region. Mobile-friendly.”*

---
## Part 6 · Insight → Decision
**What?** (điều dữ liệu cho thấy) → **So what?** (ý nghĩa với doanh nghiệp) → **Now what?** (hành động, ai làm, khi nào, đo bằng gì)
"""),
    code('''
# A simple, explainable rule: start from last Tết season's ad mix, tilt toward channels that keep more per VND
BUDGET = 2_000_000_000
tet_ads = spend[spend["month"].between("2025-12", "2026-02")].groupby("channel")["ad_spend_vnd"].sum()
mix0 = tet_ads / tet_ads.sum()
kept = result["kept per 1 VND of ads"]          # 🔧 TRY IT: use the Tết-season economics instead of the last 12 months
tilt = mix0 * kept / (mix0 * kept).sum()
plan = (tilt * BUDGET / 5e7).round() * 5e7
plan = pd.DataFrame({"last Tết mix %": (mix0 * 100).round(1), "proposed %": (plan / BUDGET * 100).round(1), "proposed (million VND)": plan / 1e6})
plan.sort_values("proposed %", ascending=False)
'''),
    code('''
from IPython.display import Markdown
memo = f"""
### 📝 Decision memo — Tết 2027 marketing (VND 2 billion)
**1. Timing.** Tết is on 06/02/2027. Gift-box demand reaches 10% by **{fc.loc[fc['cum'] >= .1, 'date'].iloc[0]:%d/%m}**, peaks around **{fc.loc[fc['units'].idxmax(), 'date']:%d/%m}**,
last shipping day **03/02**. Stock ~**{total_2027:,.0f}** boxes; never plan on calendar dates ({late:.0%} of demand would fall after the cut-off).

**2. Money.** Allocate by what we keep, not by GMV: {", ".join(f"{c} {v:.0f}%" for c, v in plan["proposed %"].sort_values(ascending=False).items())}.
Keep TikTok Shop for reach, but push **prepaid** vouchers (COD failures) and pay creators on **delivered** orders.

**3. Customers.** Re-activate last year's **Tết Gifters** and corporate buyers early (Website/Zalo); protect **Loyal Brewers** — the top 20% bring {top.head(int(len(top) * .2)).sum() / top.sum():.0%} of revenue.

**4. Operations.** Delivery complaints concentrate in the **Central** region; agree peak capacity with carriers and strengthen gift-box packaging before 01/01/2027.

**KPIs to watch weekly:** delivered revenue · failed+returned % by channel · VND kept per 1 VND of ads · gift boxes vs forecast.
"""
Markdown(memo)
'''),
    md("""
---
### Appendix · Save the clean data (for VS Code / Excel users)
"""),
    code('''
# sales.to_parquet("vietbrew_sales_clean.parquet", index=False)        # fast, small — opens in Python, Power BI
# sales.sample(100_000, random_state=1).to_csv("vietbrew_sample_100k.csv", index=False)   # small enough for Excel
'''),
]

for cell in demo.cells:
    cell.source = cell.source.replace("__DASHBOARD__", ARGS.dashboard)

for nb, name in [(hello, "00_Hello_VietBrew.ipynb"), (demo, "01_VietBrew_Live_Demo.ipynb")]:
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["colab"] = {"provenance": [], "toc_visible": True}
    nbf.write(nb, NB / name)
    print("wrote", NB / name)
