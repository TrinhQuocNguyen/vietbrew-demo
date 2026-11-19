"""
Shared analysis functions for the VietBrew demo (used by build_outputs.py, the dashboard and to cross-check
the Colab notebook). The notebook re-implements the same steps inline so learners can read them.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


# ----------------------------------------------------------------------------- loading & cleaning
def strip_accents(s: str) -> str:
    s = str(s).replace("đ", "d").replace("Đ", "D")
    return "".join(ch for ch in unicodedata.normalize("NFD", s) if unicodedata.category(ch) != "Mn")


ALIASES = {"hcm": "ho chi minh", "tphcm": "ho chi minh", "sai gon": "ho chi minh", "ho chi minh city": "ho chi minh",
           "hn": "ha noi", "hanoi": "ha noi", "tt hue": "thua thien hue", "hue": "hue",
           "vung tau": "ba ria - vung tau", "br-vt": "ba ria - vung tau", "ba ria vung tau": "ba ria - vung tau"}


def norm_province(s: str) -> str:
    s = strip_accents(s).lower().strip()
    s = re.sub(r"^(tp\.?\s*|thanh pho\s+|tinh\s+)", "", s).strip()
    return ALIASES.get(s, s)


def province_lookup(mapping: pd.DataFrame) -> dict:
    lk = {norm_province(r.old_province): r.new_province for r in mapping.itertuples()}
    lk.update({norm_province(n): n for n in mapping["new_province"].unique()})
    return lk


def load_raw(path=DATA / "raw" / "vietbrew_sales_2023_2026.csv.gz") -> pd.DataFrame:
    return pd.read_csv(path, dtype={"unit_price": str})


def clean(raw: pd.DataFrame, mapping: pd.DataFrame, tet: pd.DataFrame) -> pd.DataFrame:
    df = raw.drop_duplicates().copy()
    df["order_datetime"] = pd.to_datetime(df["order_datetime"])
    df["date"] = df["order_datetime"].dt.normalize()
    df["unit_price"] = df["unit_price"].str.replace(".", "", regex=False).astype("int64")
    df["gross_vnd"] = df["quantity"] * df["unit_price"]
    df["sales_vnd"] = df["gross_vnd"] - df["discount_vnd"]          # what the customer pays
    lk = province_lookup(mapping)
    df["province_raw"] = df["province"]
    df["province"] = df["province_raw"].map(lambda s: lk.get(norm_province(s), "UNKNOWN"))
    df["region"] = df["province"].map(dict(zip(mapping["new_province"], mapping["region"])))
    df["is_cod"] = df["payment_method"].eq("COD")
    tet_dates = pd.to_datetime(tet["tet_date"]).sort_values().to_numpy()
    d = df["date"].to_numpy()
    idx = np.searchsorted(tet_dates, d, side="left")
    df["days_to_tet"] = (tet_dates[np.clip(idx, 0, len(tet_dates) - 1)] - d).astype("timedelta64[D]").astype(int)
    df["tet_season"] = pd.DatetimeIndex(tet_dates[np.clip(idx, 0, len(tet_dates) - 1)]).year
    return df


def load_all():
    raw = load_raw()
    mapping = pd.read_csv(DATA / "province_mapping_2025.csv")
    tet = pd.read_csv(DATA / "tet_dates.csv")
    sales = clean(raw, mapping, tet)
    return {
        "raw": raw, "sales": sales, "mapping": mapping, "tet": tet,
        "products": pd.read_csv(DATA / "products.csv"),
        "costs": pd.read_csv(DATA / "channel_costs.csv"),
        "spend": pd.read_csv(DATA / "marketing_spend.csv"),
        "reviews": pd.read_csv(DATA / "reviews.csv"),
    }


# ----------------------------------------------------------------------------- metrics
def valid(sales):  # orders that were not cancelled
    return sales[sales["order_status"] != "Cancelled"]


def channel_economics(sales, products, costs, spend, start, end) -> pd.DataFrame:
    """Gross ROAS (what platforms report) vs net contribution per VND of ads (what the business keeps)."""
    s = sales[(sales["date"] >= start) & (sales["date"] <= end)]
    s = s[s["order_status"] != "Cancelled"].merge(products[["sku", "unit_cost"]], on="sku")
    s = s.merge(costs, on="channel")
    delivered = s["order_status"].eq("Delivered")
    returned = s["order_status"].eq("Returned")
    s["net_revenue"] = np.where(delivered, s["sales_vnd"], 0)
    s["fees"] = s["net_revenue"] * (s["platform_commission"] + s["affiliate_commission"]
                                    + np.where(s["is_cod"], 0, s["payment_fee_prepaid"]))
    s["cogs"] = s["unit_cost"] * s["quantity"] * np.where(delivered, 1.0, np.where(returned, s["returned_goods_write_off"], 0))
    g = s.groupby("channel").agg(gmv=("sales_vnd", "sum"), net_revenue=("net_revenue", "sum"), fees=("fees", "sum"),
                                 cogs=("cogs", "sum"), discount=("discount_vnd", "sum"), gross=("gross_vnd", "sum"))
    o = s.drop_duplicates("order_id").merge(costs[["channel"]], on="channel")
    oc = o.groupby("channel").agg(orders=("order_id", "size"),
                                  failed=("order_status", lambda x: (x == "Failed delivery").sum()),
                                  returned=("order_status", lambda x: (x == "Returned").sum()),
                                  cod=("is_cod", "mean"))
    g = g.join(oc)
    c = costs.set_index("channel")
    g["logistics"] = g["orders"] * c["fulfilment_cost_per_order"] + (g["failed"] + g["returned"]) * c["reverse_logistics_per_failed_or_returned"]
    g["contribution"] = g["net_revenue"] - g["fees"] - g["cogs"] - g["logistics"]
    m0, m1 = pd.Timestamp(start).strftime("%Y-%m"), pd.Timestamp(end).strftime("%Y-%m")
    sp = spend[(spend["month"] >= m0) & (spend["month"] <= m1)].groupby("channel")["ad_spend_vnd"].sum()
    g["ad_spend"] = sp
    g["gross_roas"] = g["gmv"] / g["ad_spend"]
    g["net_return_per_vnd"] = g["contribution"] / g["ad_spend"]
    g["profit_after_ads"] = g["contribution"] - g["ad_spend"]
    g["fail_return_rate"] = (g["failed"] + g["returned"]) / g["orders"]
    g["discount_rate"] = g["discount"] / g["gross"]
    return g.sort_values("gross_roas", ascending=False)


def customer_features(sales, ref_date) -> pd.DataFrame:
    """RFM (recency, frequency, monetary) + two behaviour signals: share of spend on Tết gift boxes,
    and the average discount the customer received."""
    s = valid(sales)
    d = s[s["order_status"] == "Delivered"]
    f = d.groupby("customer_id").agg(last=("date", "max"), frequency=("order_id", "nunique"), monetary=("sales_vnd", "sum"))
    f["recency"] = (pd.Timestamp(ref_date) - f["last"]).dt.days
    g = s.assign(gift=s["category"].eq("Tết Gift Box") * s["sales_vnd"]).groupby("customer_id")[["gift", "sales_vnd", "discount_vnd", "gross_vnd"]].sum()
    f["gift_share"] = g["gift"] / g["sales_vnd"]
    f["discount_rate"] = g["discount_vnd"] / g["gross_vnd"]
    return f.drop(columns="last").dropna()


FEATURES = ["recency", "frequency", "monetary", "gift_share", "discount_rate"]


def segment_customers(f: pd.DataFrame, k=4, seed=42) -> pd.DataFrame:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    X = f[FEATURES].copy()
    X["frequency"] = np.log1p(X["frequency"])
    X["monetary"] = np.log1p(X["monetary"])
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(StandardScaler().fit_transform(X))
    out = f.copy()
    out["cluster"] = km.labels_
    return name_segments(out)


def name_segments(seg: pd.DataFrame) -> pd.DataFrame:
    """Name clusters with simple business rules on the cluster averages."""
    prof = seg.groupby("cluster")[FEATURES].mean()
    names = {}
    for col, name, how in [("frequency", "Loyal Brewers", "max"), ("gift_share", "Tết Gifters", "max"),
                           ("discount_rate", "Deal Hunters", "max"), ("frequency", "One-time Triers", "min")]:
        rest = prof.drop(index=list(names))
        names[getattr(rest[col], "idx" + how)()] = name
    seg["segment"] = seg["cluster"].map(names)
    return seg


def segment_profile(seg: pd.DataFrame) -> pd.DataFrame:
    p = seg.groupby("segment").agg(customers=("recency", "size"), recency=("recency", "mean"), frequency=("frequency", "mean"),
                                   monetary=("monetary", "mean"), gift_share=("gift_share", "mean"),
                                   discount_rate=("discount_rate", "mean"), revenue=("monetary", "sum"))
    p["customer_share"] = p["customers"] / p["customers"].sum()
    p["revenue_share"] = p["revenue"] / p["revenue"].sum()
    order = ["Loyal Brewers", "Tết Gifters", "Deal Hunters", "One-time Triers"]
    return p.reindex(order)


# ----------------------------------------------------------------------------- Tết alignment & forecast
def gift_daily(sales) -> pd.DataFrame:
    s = valid(sales)
    s = s[s["category"].eq("Tết Gift Box")]
    return s.groupby(["tet_season", "days_to_tet", "date"], as_index=False).agg(units=("quantity", "sum"), revenue=("sales_vnd", "sum"))


def tet_forecast(sales, tet, target_season=2027, window=60, last_ship=3):
    """Average shape (share of season units by days before Tết) x season total grown at the recent growth rate."""
    g = gift_daily(sales)
    g = g[(g["days_to_tet"] <= window)]
    full = [y for y in g["tet_season"].unique() if y < target_season and y >= 2024]   # 2023 season is cut by the data start
    tot = g[g["tet_season"].isin(full)].groupby("tet_season")["units"].sum()
    growth = (tot / tot.shift(1)).dropna()
    g_rate = float(np.exp(np.log(growth).mean()))
    shape = (g[g["tet_season"].isin(full)].groupby(["tet_season", "days_to_tet"])["units"].sum()
             .groupby(level=0).transform(lambda x: x / x.sum()).groupby(level=1).mean())
    shape = shape.reindex(range(window + 1), fill_value=0)
    shape = shape / shape.sum()
    total = float(tot.loc[max(full)] * g_rate)
    tet_date = pd.Timestamp(tet.set_index("tet_year").loc[target_season, "tet_date"])
    fc = pd.DataFrame({"days_to_tet": shape.index, "share": shape.values})
    fc["date"] = tet_date - pd.to_timedelta(fc["days_to_tet"], unit="D")
    fc["units"] = fc["share"] * total
    fc = fc.sort_values("date").reset_index(drop=True)
    fc["cum_share"] = fc["share"].cumsum()
    # naive: copy last season's units onto the same CALENDAR dates
    prev = g[g["tet_season"] == max(full)].copy()
    prev["date_next_year"] = prev["date"] + pd.DateOffset(years=1)
    naive_after_cutoff = prev.loc[prev["date_next_year"] > tet_date - pd.Timedelta(days=last_ship), "units"].sum() / prev["units"].sum()
    return {"forecast": fc, "season_totals": tot, "growth": g_rate, "total": total, "tet_date": tet_date,
            "naive_share_after_cutoff": float(naive_after_cutoff), "prev_season": int(max(full))}
