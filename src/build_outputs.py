"""
Build every derived artefact from the raw VietBrew data:
  data/clean/vietbrew_sales_clean.parquet      cleaned sales lines (for VS Code / advanced users)
  dashboard/data/*.parquet                      small aggregates for the Streamlit dashboard
  slides/_build/slide_data.json                 every number shown on the slides (single source of truth)

Run:  python src/build_outputs.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from analysis import (DATA, ROOT, channel_economics, customer_features, load_all, segment_customers,  # noqa: E402
                      segment_profile, strip_accents, tet_forecast, valid, gift_daily)

DASH = ROOT / "dashboard" / "data"
SLD = ROOT / "slides" / "_build"
for d in (DATA / "clean", DASH, SLD):
    d.mkdir(parents=True, exist_ok=True)

A = load_all()
raw, s, products, costs, spend, reviews, tet = (A[k] for k in ["raw", "sales", "products", "costs", "spend", "reviews", "tet"])
REF = pd.Timestamp("2026-11-30")
out: dict = {}

# ------------------------------------------------------------------ 1. dataset facts
o = s.drop_duplicates("order_id")
out["dataset"] = {
    "raw_rows": int(len(raw)), "clean_rows": int(len(s)), "duplicates_removed": int(len(raw) - len(s)),
    "orders": int(s["order_id"].nunique()), "customers": int(s["customer_id"].nunique()),
    "date_min": s["date"].min().strftime("%Y-%m-%d"), "date_max": s["date"].max().strftime("%Y-%m-%d"),
    "excel_row_limit": 1048576, "raw_file_mb": round((DATA / "raw" / "vietbrew_sales_2023_2026.csv.gz").stat().st_size / 1e6, 1),
    "skus": int(products.shape[0]), "channels": sorted(s["channel"].unique().tolist()),
    "province_spellings": int(s["province_raw"].nunique()), "provinces_clean": int(s["province"].nunique()),
    "revenue_total_bn": round(valid(s)["sales_vnd"].sum() / 1e9, 1),
}

# ------------------------------------------------------------------ 2. cleaning traps
naive = pd.to_numeric(raw["unit_price"], errors="coerce")
out["price_trap"] = {
    "example_text": "145.000", "naive_value": 145.0, "correct_value": 145000,
    "naive_unparsed_rows": int(naive.isna().sum()),
    "naive_revenue_bn": round(float((raw["quantity"] * naive.fillna(0)).sum() / 1e9), 4),
    "correct_revenue_bn": round(float(s["gross_vnd"].sum() / 1e9), 1),
}
bd = s[s["province_raw"].map(lambda x: strip_accents(x).lower()).str.contains("binh duong")]
bd_m = bd.groupby(bd["date"].dt.to_period("M"))["order_id"].nunique()
hcm_variants = s.loc[s["province"].eq("TP. Hồ Chí Minh"), "province_raw"].value_counts()
out["province_trap"] = {
    "binh_duong_orders_jun_2025": int(bd_m.get(pd.Period("2025-06"), 0)),
    "binh_duong_orders_aug_2025": int(bd_m.get(pd.Period("2025-08"), 0)),
    "hcm_spellings": hcm_variants.head(12).index.tolist(), "hcm_spelling_count": int(len(hcm_variants)),
}

# ------------------------------------------------------------------ 3. monthly revenue & YoY trap
v = valid(s)
mon = v.groupby(v["date"].dt.to_period("M"))["sales_vnd"].sum() / 1e9
out["monthly_bn"] = {str(k): round(float(x), 2) for k, x in mon.items()}
out["yoy"] = {
    "feb_2026": round(float(mon[pd.Period("2026-02")]), 1), "feb_2025": round(float(mon[pd.Period("2025-02")]), 1),
    "feb_ratio": round(float(mon[pd.Period("2026-02")] / mon[pd.Period("2025-02")]), 1),
    "jan_2026": round(float(mon[pd.Period("2026-01")]), 1), "jan_2025": round(float(mon[pd.Period("2025-01")]), 1),
    "jan_change_pct": round(float((mon[pd.Period("2026-01")] / mon[pd.Period("2025-01")] - 1) * 100), 1),
    "season_2025_bn": round(float(v.loc[v["tet_season"].eq(2025) & v["days_to_tet"].between(0, 60), "sales_vnd"].sum() / 1e9), 1),
    "season_2026_bn": round(float(v.loc[v["tet_season"].eq(2026) & v["days_to_tet"].between(0, 60), "sales_vnd"].sum() / 1e9), 1),
}
out["yoy"]["season_growth_pct"] = round((out["yoy"]["season_2026_bn"] / out["yoy"]["season_2025_bn"] - 1) * 100, 1)
yr = v.groupby(v["date"].dt.year)["sales_vnd"].sum() / 1e9
out["annual_bn"] = {int(k): round(float(x), 1) for k, x in yr.items()}

# ------------------------------------------------------------------ 4. channel mix
mix = v.pivot_table(index=v["date"].dt.year, columns="channel", values="sales_vnd", aggfunc="sum")
mix = mix.div(mix.sum(axis=1), axis=0) * 100
out["channel_share_pct"] = {ch: {int(y): round(float(mix.loc[y, ch]), 1) for y in mix.index} for ch in mix.columns}
o26 = o[o["date"].dt.year.eq(2026)]
st = pd.crosstab(o26["channel"], o26["order_status"], normalize="index") * 100
out["status_2026_pct"] = {ch: {k: round(float(st.loc[ch, k]), 1) for k in st.columns} for ch in st.index}
out["cod_2026_pct"] = {ch: round(float(x * 100), 1) for ch, x in o26.groupby("channel")["is_cod"].mean().items()}

# ------------------------------------------------------------------ 5. Tết alignment
g = gift_daily(s)
aligned = {}
for y in [2024, 2025, 2026]:
    a = g[(g["tet_season"] == y) & (g["days_to_tet"] <= 60)].groupby("days_to_tet")["units"].sum().reindex(range(61), fill_value=0)
    aligned[y] = a.rolling(5, center=True, min_periods=1).mean().round(1)
out["tet_aligned_units"] = {str(y): aligned[y].tolist()[::-1] for y in aligned}           # index 0 = 60 days before Tết
greg = {}
for y in [2024, 2025, 2026]:
    d0, d1 = pd.Timestamp(f"{y - 1}-12-01"), pd.Timestamp(f"{y}-02-28")
    a = g[(g["date"] >= d0) & (g["date"] <= d1)].groupby("date")["units"].sum().reindex(pd.date_range(d0, d1), fill_value=0)
    greg[y] = a.rolling(5, center=True, min_periods=1).mean().round(1).tolist()[:90]
out["tet_gregorian_units"] = {str(y): greg[y] for y in greg}
out["tet_dates"] = dict(zip(tet["tet_year"].astype(str), tet["tet_date"]))
peak = {y: int(aligned[y].idxmax()) for y in aligned}
out["tet_peak_days_before"] = {str(y): peak[y] for y in peak}

F = tet_forecast(s, tet)
fc = F["forecast"]
first = fc.loc[fc["cum_share"] >= 0.10, "date"].iloc[0]
p80 = fc.loc[fc["cum_share"] >= 0.80, "date"].iloc[0]
pk = fc.loc[fc["units"].idxmax(), "date"]
out["forecast_2027"] = {
    "tet_date": F["tet_date"].strftime("%Y-%m-%d"), "season_units": {int(k): int(x) for k, x in F["season_totals"].items()},
    "growth_rate_pct": round((F["growth"] - 1) * 100, 1), "units_2027": int(round(F["total"], -2)),
    "date_10pct": first.strftime("%Y-%m-%d"), "date_peak": pk.strftime("%Y-%m-%d"), "date_80pct": p80.strftime("%Y-%m-%d"),
    "last_ship": (F["tet_date"] - pd.Timedelta(days=3)).strftime("%Y-%m-%d"),
    "naive_share_after_cutoff_pct": round(F["naive_share_after_cutoff"] * 100, 1),
    "units_by_week": None,
}
wk = fc.assign(week=fc["date"].dt.to_period("W-SUN").dt.start_time).groupby("week")["units"].sum()
out["forecast_2027"]["units_by_week"] = {k.strftime("%Y-%m-%d"): int(round(x, -1)) for k, x in wk.items()}

# ------------------------------------------------------------------ 6. customers
feat = customer_features(s, REF)
seg = segment_customers(feat)
prof = segment_profile(seg)
mega = v["date"].dt.month.eq(v["date"].dt.day)
cust_ch = v.assign(mega=mega).merge(seg[["segment"]], left_on="customer_id", right_index=True)
ch_mix = pd.crosstab(cust_ch["segment"], cust_ch["channel"], values=cust_ch["sales_vnd"], aggfunc="sum", normalize="index") * 100
out["segments"] = {}
for name, r in prof.iterrows():
    out["segments"][name] = {
        "customers": int(r["customers"]), "customer_share_pct": round(r["customer_share"] * 100, 1),
        "revenue_share_pct": round(r["revenue_share"] * 100, 1), "orders_avg": round(r["frequency"], 1),
        "spend_avg_k": int(round(r["monetary"] / 1000, -1)), "recency_days": int(r["recency"]),
        "gift_share_pct": round(r["gift_share"] * 100, 1), "discount_pct": round(r["discount_rate"] * 100, 1),
        "top_channel": ch_mix.loc[name].idxmax(), "top_channel_pct": round(float(ch_mix.loc[name].max()), 1),
    }
m = feat["monetary"].sort_values(ascending=False)
cum = m.cumsum() / m.sum()
out["pareto"] = {"top20_share_pct": round(float(cum.iloc[int(len(m) * 0.2) - 1] * 100), 1),
                 "top5_share_pct": round(float(cum.iloc[int(len(m) * 0.05) - 1] * 100), 1),
                 "customers_with_delivery": int(len(m))}
truth_p = DATA / "instructor" / "customers_truth.csv"
if truth_p.exists():
    truth = pd.read_csv(truth_p).set_index("customer_id")["segment"].reindex(seg.index)
    ct = pd.crosstab(seg["cluster"], truth)
    out["segments_purity_pct"] = round(float(ct.max(axis=1).sum() / ct.values.sum() * 100), 1)

# ------------------------------------------------------------------ 7. channel economics: gross ROAS vs net return
def econ(start, end):
    e = channel_economics(s, products, costs, spend, start, end)
    return {ch: {"gmv_bn": round(r["gmv"] / 1e9, 2), "ad_spend_bn": round(r["ad_spend"] / 1e9, 2),
                 "gross_roas": round(r["gross_roas"], 1), "net_return_per_vnd": round(r["net_return_per_vnd"], 2),
                 "profit_after_ads_bn": round(r["profit_after_ads"] / 1e9, 2), "fail_return_pct": round(r["fail_return_rate"] * 100, 1),
                 "discount_pct": round(r["discount_rate"] * 100, 1), "cod_pct": round(r["cod"] * 100, 1),
                 "contribution_bn": round(r["contribution"] / 1e9, 2)} for ch, r in e.iterrows()}


out["econ_12m"] = econ("2025-12-01", "2026-11-30")
out["econ_tet2026"] = econ("2025-12-01", "2026-02-28")

# budget allocation for Tết 2027: start from last Tết season's mix, tilt by net return, round to 50 M VND
BUDGET = 2_000_000_000
e = pd.DataFrame(out["econ_12m"]).T
tet_ads = spend[spend["month"].between("2025-12", "2026-02")].groupby("channel")["ad_spend_vnd"].sum()
mix0 = (tet_ads / tet_ads.sum()).reindex(e.index)
tilt = mix0 * e["net_return_per_vnd"] / (mix0 * e["net_return_per_vnd"]).sum()
alloc = (tilt / tilt.sum() * BUDGET / 5e7).round() * 5e7
alloc.iloc[np.argmax(alloc.values)] += BUDGET - alloc.sum()
exp_contrib = (alloc * e["net_return_per_vnd"]).sum()
old_contrib = (mix0 * BUDGET * e["net_return_per_vnd"]).sum()
out["budget_2027"] = {
    "total_bn": 2.0,
    "last_season_mix_pct": {ch: round(float(x * 100), 1) for ch, x in mix0.items()},
    "proposed_bn": {ch: round(float(x / 1e9), 2) for ch, x in alloc.items()},
    "proposed_pct": {ch: round(float(x / BUDGET * 100), 1) for ch, x in alloc.items()},
    "expected_contribution_bn": round(float(exp_contrib / 1e9), 2),
    "same_mix_contribution_bn": round(float(old_contrib / 1e9), 2),
}
out["budget_2027"]["uplift_bn"] = round(out["budget_2027"]["expected_contribution_bn"] - out["budget_2027"]["same_mix_contribution_bn"], 2)

# ------------------------------------------------------------------ 8. reviews (rule-based baseline + reference labels)
ALIASES = {"hcm": "ho chi minh", "tphcm": "ho chi minh", "sai gon": "ho chi minh", "ho chi minh city": "ho chi minh",
           "hn": "ha noi", "hanoi": "ha noi", "tt hue": "thua thien hue", "vung tau": "ba ria - vung tau",
           "br-vt": "ba ria - vung tau", "ba ria vung tau": "ba ria - vung tau"}


def norm(s):  # identical to the notebook
    s = strip_accents(str(s)).lower().strip()
    s = re.sub(r"^(tp\.?\s*|thanh pho\s+|tinh\s+)", "", s).strip()
    return ALIASES.get(s, s)


TEEN = [(r"\b(k|ko|hong|hok)\b", "khong"), (r"\b(dc|dk)\b", "duoc"), (r"\b(cf|cafe)\b", "ca phe"), (r"\b(r|rui)\b", "roi")]
RULES = [("Service", ["sai san pham", "giao hop", "thieu hang", "tu van", "tra loi", "doi hang"]),
         ("Packaging", ["mop", "dong goi", "bao bi", "rach", "bi be", "thung", "chong soc", "goi hang", "trau xuoc", "bien dang"]),
         ("Delivery", ["giao", "ship", "tre", "cham", "kho trung chuyen", "dung hen", "nhan duoc", "co hang"]),
         ("Price", ["gia", "sale", "hoi dat", "giam sau"]),
         ("Taste", ["vi ", "thom", "ngon", "chua", "khet", "nhat", "gu", "dam", "uong"])]
NEG = ["cham", "lau", "tre", "mop", "be", "rach", "that vong", "sai", "thieu", "nhat", "khet", "gat", "hoi dat", "uot",
       "bien dang", "trau xuoc", "so sai", "khong ai", "khong thay", "khong giong", "khong hop", "xem lai", "cho dai", "bi chua"]
POS = ["nhanh", "dep", "ngon", "thom", "ok", "chu dao", "nhiet tinh", "qua hoi", "5 sao", "dang tien", "hop ly", "chac chan",
       "can than", "mua lai", "dung hen", "than thien", "thich", "dung gu", "ung ho", "dam da"]


def classify(text):  # identical to rule_classify() in the notebook
    t = norm(text)
    t = re.sub(r"(\w)\1{2,}", r"\1", t)
    for a, b in TEEN:
        t = re.sub(a, b, t)
    aspect = next((a for a, kws in RULES if any(k in t for k in kws)), "Other")
    neg, pos = sum(k in t for k in NEG), sum(k in t for k in POS)
    return aspect, "Negative" if neg > pos else "Positive" if pos > neg else "Neutral"


rv = reviews.copy()
rv[["rule_aspect", "rule_sentiment"]] = rv["review_text"].apply(lambda t: pd.Series(classify(t)))
lab_p = DATA / "instructor" / "reviews_labels.csv"
if lab_p.exists():
    rv = rv.merge(pd.read_csv(lab_p), on="review_id")
    acc_aspect = (rv["rule_aspect"] == rv["aspect"]).mean()
    acc_sent = (rv["rule_sentiment"] == rv["sentiment"]).mean()
    acc_both = ((rv["rule_aspect"] == rv["aspect"]) & (rv["rule_sentiment"] == rv["sentiment"])).mean()
    out["reviews_rule_accuracy_pct"] = {"aspect": round(acc_aspect * 100, 1), "sentiment": round(acc_sent * 100, 1), "both": round(acc_both * 100, 1)}
    neg = rv[rv["sentiment"] == "Negative"]
    out["reviews"] = {
        "total": int(len(rv)), "negative": int(len(neg)),
        "negative_by_aspect": neg["aspect"].value_counts().to_dict(),
        "negative_by_aspect_region": {a: neg[neg["aspect"] == a]["region"].value_counts().reindex(["North", "Central", "South"], fill_value=0).to_dict()
                                      for a in ["Delivery", "Packaging", "Service", "Taste", "Price"]},
        "positive_by_aspect": rv[rv["sentiment"] == "Positive"]["aspect"].value_counts().to_dict(),
        "ops_share_of_negative_pct": round(float(neg["aspect"].isin(["Delivery", "Packaging", "Service"]).mean() * 100), 1),
        "central_share_of_delivery_neg_pct": round(float((neg.loc[neg["aspect"] == "Delivery", "region"] == "Central").mean() * 100), 1),
        "central_share_of_customers_pct": round(float((s.drop_duplicates("customer_id")["region"] == "Central").mean() * 100), 1),
        "teen_examples": rv.loc[rv["review_text"].str.contains(r"\b(?:k|ko|hông|dc|đc|cf|wá)\b|hông|nhanhhh|thơmmm", regex=True), "review_text"].head(40).tolist(),
    }
rn = rv[rv["rule_sentiment"] == "Negative"]
out["reviews_rules"] = {
    "negative": int(len(rn)), "negative_by_aspect": rn["rule_aspect"].value_counts().to_dict(),
    "negative_by_aspect_region": {a: rn[rn["rule_aspect"] == a]["region"].value_counts().reindex(["North", "Central", "South"], fill_value=0).to_dict()
                                  for a in ["Delivery", "Packaging", "Service", "Taste", "Price"]},
    "ops_share_of_negative_pct": round(float(rn["rule_aspect"].isin(["Delivery", "Packaging", "Service"]).mean() * 100), 1),
    "central_share_of_delivery_neg_pct": round(float((rn.loc[rn["rule_aspect"] == "Delivery", "region"] == "Central").mean() * 100), 1),
}
print("rules:", out["reviews_rules"])
rv.to_parquet(DASH / "reviews.parquet", index=False)

# ------------------------------------------------------------------ 9. dashboard aggregates
s2 = s.merge(products[["sku", "unit_cost"]], on="sku")
s2["cogs_vnd"] = s2["unit_cost"] * s2["quantity"]
daily = s2.groupby(["date", "channel", "region", "category", "order_status", "is_cod"], as_index=False).agg(
    gross_vnd=("gross_vnd", "sum"), discount_vnd=("discount_vnd", "sum"), sales_vnd=("sales_vnd", "sum"),
    cogs_vnd=("cogs_vnd", "sum"), units=("quantity", "sum"))
daily.to_parquet(DASH / "daily_sales.parquet", index=False)
od = s.drop_duplicates("order_id").groupby(["date", "channel", "region", "order_status", "is_cod"], as_index=False).agg(orders=("order_id", "size"))
od.to_parquet(DASH / "daily_orders.parquet", index=False)
seg.reset_index()[["customer_id", "segment", "recency", "frequency", "monetary", "gift_share", "discount_rate"]].to_parquet(DASH / "customer_segments.parquet", index=False)
fc.to_parquet(DASH / "tet_forecast_2027.parquet", index=False)
g.to_parquet(DASH / "gift_daily.parquet", index=False)
for f in ["channel_costs.csv", "marketing_spend.csv", "tet_dates.csv", "products.csv"]:
    pd.read_csv(DATA / f).to_csv(DASH / f, index=False)

# clean sales for VS Code users
s.drop(columns=["province_raw"]).to_parquet(DATA / "clean" / "vietbrew_sales_clean.parquet", index=False)

# extra facts for slides
oo = s.drop_duplicates("order_id")
oo = oo[oo["order_status"] != "Cancelled"]
peak = oo["days_to_tet"].between(3, 28)
fr = lambda m: round(float((oo.loc[m, "order_status"] == "Failed delivery").mean() * 100), 1)
out["central_fail_pct"] = {"central_peak": fr(peak & oo["region"].eq("Central")), "central_offpeak": fr(~peak & oo["region"].eq("Central")),
                           "other_peak": fr(peak & ~oo["region"].eq("Central"))}
ex = {}
for asp, sen, pat in [("Delivery", "Negative", r"hông|ship|quá trời|k |ko "), ("Packaging", "Negative", r"^[a-z ,]+$|móp"),
                      ("Taste", "Positive", r"cf|thơmmm|ngonnn|cafe"), ("Price", "Negative", r"^k |hông sale|ko sale"),
                      ("Service", "Negative", r"hông|nhắn")]:
    cand = rv[(rv["aspect"] == asp) & (rv["sentiment"] == sen) & rv["review_text"].str.contains(pat, regex=True)]
    if len(cand):
        r = cand.iloc[0]
        ex[asp] = {"text": r["review_text"], "sentiment": sen, "rule_aspect": r["rule_aspect"], "rule_sentiment": r["rule_sentiment"]}
out["review_examples"] = ex
out["app_py_lines"] = sum(1 for _ in open(ROOT / "dashboard" / "app.py", encoding="utf-8"))
(SLD / "slide_data.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
print(json.dumps({k: out[k] for k in ["dataset", "yoy", "forecast_2027", "pareto", "budget_2027"]}, ensure_ascii=False, indent=1, default=str))
print(json.dumps(out["segments"], ensure_ascii=False, indent=1))
print(json.dumps(out["econ_12m"], ensure_ascii=False, indent=1))
print(json.dumps(out["econ_tet2026"], ensure_ascii=False, indent=1))
print(out.get("reviews_rule_accuracy_pct"), out.get("segments_purity_pct"))
print({k: out["reviews"][k] for k in ["negative_by_aspect", "negative_by_aspect_region", "ops_share_of_negative_pct", "central_share_of_delivery_neg_pct", "central_share_of_customers_pct"]})
print(out["province_trap"], out["price_trap"], out["tet_peak_days_before"])
