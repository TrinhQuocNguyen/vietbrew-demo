"""
VietBrew Tết 2027 dashboard — Streamlit app for the upGrad x GGU MBA Bilingual demo class.
VietBrew is a FICTIONAL company; all data are SIMULATED.

Local:   streamlit run dashboard/app.py
Cloud:   Streamlit Community Cloud → New app → main file: dashboard/app.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

D = Path(__file__).parent / "data"
CH_COLORS = {"Shopee": "#EE4D2D", "TikTok Shop": "#111111", "Lazada": "#1A3C6E", "Website": "#F4831F"}
SEG_ORDER = ["Loyal Brewers", "Tết Gifters", "Deal Hunters", "One-time Triers"]

st.set_page_config(page_title="VietBrew · Tết 2027 dashboard", page_icon="☕", layout="wide")


@st.cache_data
def load():
    daily = pd.read_parquet(D / "daily_sales.parquet")
    orders = pd.read_parquet(D / "daily_orders.parquet")
    for df in (daily, orders):
        df["date"] = pd.to_datetime(df["date"])
    return {
        "daily": daily, "orders": orders,
        "gift": pd.read_parquet(D / "gift_daily.parquet"),
        "fc": pd.read_parquet(D / "tet_forecast_2027.parquet"),
        "seg": pd.read_parquet(D / "customer_segments.parquet"),
        "rev": pd.read_parquet(D / "reviews.parquet"),
        "costs": pd.read_csv(D / "channel_costs.csv").set_index("channel"),
        "spend": pd.read_csv(D / "marketing_spend.csv"),
    }


A = load()
daily, orders = A["daily"], A["orders"]

# ------------------------------------------------------------------ sidebar
st.sidebar.markdown("### ☕ VietBrew")
st.sidebar.caption("Fictional company · simulated data\nupGrad × GGU MBA Bilingual — demo class")
dmin, dmax = daily["date"].min().date(), daily["date"].max().date()
start, end = st.sidebar.date_input("Period", (pd.Timestamp("2025-12-01").date(), dmax), min_value=dmin, max_value=dmax)
channels = st.sidebar.multiselect("Channels", list(CH_COLORS), default=list(CH_COLORS))
regions = st.sidebar.multiselect("Regions", ["North", "Central", "South"], default=["North", "Central", "South"])
start, end = pd.Timestamp(start), pd.Timestamp(end)


def filt(df):
    return df[(df["date"] >= start) & (df["date"] <= end) & df["channel"].isin(channels) & df["region"].isin(regions)]


def prev(df):
    span = end - start
    p0, p1 = start - span - pd.Timedelta(days=1), start - pd.Timedelta(days=1)
    return df[(df["date"] >= p0) & (df["date"] <= p1) & df["channel"].isin(channels) & df["region"].isin(regions)]


def kpis(ds, do):
    ok = ds[ds["order_status"] != "Cancelled"]
    delivered = ds[ds["order_status"] == "Delivered"]["sales_vnd"].sum()
    n = do[do["order_status"] != "Cancelled"]["orders"].sum()
    bad = do[do["order_status"].isin(["Failed delivery", "Returned"])]["orders"].sum()
    cod = do[(do["order_status"] != "Cancelled") & do["is_cod"]]["orders"].sum()
    return {"gmv": ok["sales_vnd"].sum(), "delivered": delivered, "orders": n,
            "bad": bad / n if n else 0, "cod": cod / n if n else 0}


st.title("VietBrew · From data to decision")
st.caption(f"Period {start:%d/%m/%Y} – {end:%d/%m/%Y} · {', '.join(channels)} · {', '.join(regions)}")

t1, t2, t3, t4, t5 = st.tabs(["📈 Overview", "🧧 Tết 2027 planner", "💰 Channels", "👥 Customers", "💬 Reviews"])

# ------------------------------------------------------------------ overview
with t1:
    k, kp = kpis(filt(daily), filt(orders)), kpis(prev(daily), prev(orders))

    def delta(a, b, pct=False):
        if not b:
            return None
        return f"{(a - b) * 100:+.1f} pts" if pct else f"{a / b - 1:+.0%}"

    c = st.columns(5)
    c[0].metric("GMV (bn VND)", f"{k['gmv'] / 1e9:,.1f}", delta(k["gmv"], kp["gmv"]))
    c[1].metric("Delivered revenue (bn VND)", f"{k['delivered'] / 1e9:,.1f}", delta(k["delivered"], kp["delivered"]))
    c[2].metric("Orders", f"{k['orders']:,.0f}", delta(k["orders"], kp["orders"]))
    c[3].metric("Failed + returned", f"{k['bad']:.1%}", delta(k["bad"], kp["bad"], True), delta_color="inverse")
    c[4].metric("Cash on delivery", f"{k['cod']:.1%}", delta(k["cod"], kp["cod"], True), delta_color="inverse")
    st.caption("Deltas compare with the previous period of the same length. Note: in Vietnam, calendar periods shift with Tết.")

    m = daily[daily["channel"].isin(channels) & daily["region"].isin(regions) & (daily["order_status"] != "Cancelled")]
    m = m.groupby([m["date"].dt.to_period("M").astype(str), "channel"])["sales_vnd"].sum().div(1e9).reset_index()
    fig = px.bar(m, x="date", y="sales_vnd", color="channel", color_discrete_map=CH_COLORS,
                 labels={"sales_vnd": "billion VND", "date": ""}, title="Revenue per month by channel (all years)")
    fig.update_layout(height=380, legend_orientation="h", legend_y=-0.2, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, width="stretch")

    by_cat = filt(daily)
    by_cat = by_cat[by_cat["order_status"] != "Cancelled"].groupby("category")["sales_vnd"].sum().div(1e9).sort_values()
    fig = px.bar(by_cat, orientation="h", title="Revenue by category in the period (bn VND)", labels={"value": "", "category": ""})
    fig.update_layout(height=300, showlegend=False, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------------ Tết planner
with t2:
    g, fc = A["gift"], A["fc"]
    tet_date = fc["date"].max()
    first = fc.loc[fc["cum_share"] >= 0.10, "date"].iloc[0]
    p80 = fc.loc[fc["cum_share"] >= 0.80, "date"].iloc[0]
    pk = fc.loc[fc["units"].idxmax(), "date"]
    c = st.columns(4)
    c[0].metric("Forecast gift boxes", f"{fc['units'].sum():,.0f}")
    c[1].metric("10% sold by", f"{first:%d/%m/%Y}")
    c[2].metric("Peak day", f"{pk:%d/%m/%Y}")
    c[3].metric("Last shipping day", f"{tet_date - pd.Timedelta(days=3):%d/%m/%Y}")

    view = st.radio("View", ["Aligned on Tết (days before)", "Calendar dates"], horizontal=True)
    sel = g[g["tet_season"].isin([2024, 2025, 2026]) & (g["days_to_tet"] <= 60)]
    sel = sel.groupby(["tet_season", "days_to_tet", "date"], as_index=False)["units"].sum()
    if view.startswith("Aligned"):
        fig = px.line(sel, x="days_to_tet", y="units", color=sel["tet_season"].astype(str),
                      labels={"days_to_tet": "days before Tết", "units": "boxes / day", "color": "season"})
        fig.add_scatter(x=fc["days_to_tet"], y=fc["units"], name="2027 forecast", line=dict(dash="dash", width=3, color="#F4831F"))
        fig.update_xaxes(autorange="reversed")
    else:
        sel["day"] = (sel["date"] - pd.to_datetime((sel["tet_season"] - 1).astype(str) + "-12-01")).dt.days
        fig = px.line(sel, x="day", y="units", color=sel["tet_season"].astype(str),
                      labels={"day": "days since 1 December", "units": "boxes / day", "color": "season"})
    fig.update_layout(height=400, legend_orientation="h", legend_y=-0.25, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, width="stretch")
    wk = fc.groupby(fc["date"].dt.to_period("W-SUN").dt.start_time)["units"].sum().round(-1)
    st.markdown("**Weekly forecast, Tết 2027 (6/2/2027)**")
    st.dataframe(wk.rename("gift boxes").rename_axis("week starting").to_frame().T.rename(columns=lambda x: f"{x:%d/%m}"),
                 width="stretch")

# ------------------------------------------------------------------ channels
with t3:
    ds = filt(daily)
    ds = ds[ds["order_status"] != "Cancelled"].join(A["costs"], on="channel")
    dl = ds["order_status"].eq("Delivered")
    rt = ds["order_status"].eq("Returned")
    ds["net"] = np.where(dl, ds["sales_vnd"], 0)
    ds["fees"] = ds["net"] * (ds["platform_commission"] + ds["affiliate_commission"] + np.where(ds["is_cod"], 0, ds["payment_fee_prepaid"]))
    ds["cogs"] = ds["cogs_vnd"] * np.where(dl, 1.0, np.where(rt, ds["returned_goods_write_off"], 0))
    e = ds.groupby("channel")[["sales_vnd", "net", "fees", "cogs"]].sum()
    do = filt(orders)
    do = do[do["order_status"] != "Cancelled"]
    e["orders"] = do.groupby("channel")["orders"].sum()
    e["bad"] = do[do["order_status"].isin(["Failed delivery", "Returned"])].groupby("channel")["orders"].sum()
    cst = A["costs"].reindex(e.index)
    e["logistics"] = e["orders"] * cst["fulfilment_cost_per_order"] + e["bad"] * cst["reverse_logistics_per_failed_or_returned"]
    e["contribution"] = e["net"] - e["fees"] - e["cogs"] - e["logistics"]
    sp = A["spend"]
    sp = sp[(sp["month"] >= start.strftime("%Y-%m")) & (sp["month"] <= end.strftime("%Y-%m"))].groupby("channel")["ad_spend_vnd"].sum()
    if regions != ["North", "Central", "South"]:
        st.info("Ad spend is recorded per channel, not per region: ROAS below uses national ad spend — select all regions for a fair comparison.")
    e["ad_spend"] = sp
    e["Gross ROAS"] = e["sales_vnd"] / e["ad_spend"]
    e["VND kept per 1 VND of ads"] = e["contribution"] / e["ad_spend"]
    e["Failed + returned %"] = e["bad"] / e["orders"] * 100
    e["Profit after ads (bn)"] = (e["contribution"] - e["ad_spend"]) / 1e9
    e = e.sort_values("Gross ROAS", ascending=False)
    fig = go.Figure([go.Bar(name="Gross ROAS (platform view)", x=e.index, y=e["Gross ROAS"], marker_color="#8FA9CC"),
                     go.Bar(name="VND kept per 1 VND of ads", x=e.index, y=e["VND kept per 1 VND of ads"], marker_color="#F4831F")])
    fig.update_layout(barmode="group", height=380, title="The ranking flips when you count what the business keeps",
                      legend_orientation="h", legend_y=-0.15, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, width="stretch")
    st.dataframe(e[["Gross ROAS", "VND kept per 1 VND of ads", "Failed + returned %", "Profit after ads (bn)"]].round(2), width="stretch")
    st.caption("Contribution = delivered revenue − platform & creator fees − cost of goods − logistics. Cost assumptions: channel_costs.csv (illustrative).")

# ------------------------------------------------------------------ customers
with t4:
    seg = A["seg"]
    p = seg.groupby("segment").agg(customers=("customer_id", "size"), revenue=("monetary", "sum"), orders=("frequency", "mean"),
                                   spend=("monetary", "mean"), gift=("gift_share", "mean"), discount=("discount_rate", "mean")).reindex(SEG_ORDER)
    p["% customers"] = p["customers"] / p["customers"].sum() * 100
    p["% revenue"] = p["revenue"] / p["revenue"].sum() * 100
    fig = go.Figure([go.Bar(name="% of customers", x=p.index, y=p["% customers"], marker_color="#8FA9CC"),
                     go.Bar(name="% of revenue", x=p.index, y=p["% revenue"], marker_color="#1A3C6E")])
    fig.update_layout(barmode="group", height=360, title="Four segments (RFM + behaviour, K-Means)", legend_orientation="h", legend_y=-0.15,
                      margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, width="stretch")
    show = p[["customers", "% customers", "% revenue", "orders", "spend", "gift", "discount"]].rename(
        columns={"orders": "avg orders", "spend": "avg spend (VND)", "gift": "gift-box share", "discount": "avg discount"})
    st.dataframe(show.style.format({"customers": "{:,.0f}", "% customers": "{:.1f}", "% revenue": "{:.1f}", "avg orders": "{:.1f}",
                                    "avg spend (VND)": "{:,.0f}", "gift-box share": "{:.0%}", "avg discount": "{:.0%}"}), width="stretch")

# ------------------------------------------------------------------ reviews
with t5:
    rv = A["rev"]
    rv = rv[rv["channel"].isin(channels) & rv["region"].isin(regions)]
    neg = rv[rv["rule_sentiment"] == "Negative"]
    tab = pd.crosstab(neg["rule_aspect"], neg["region"]).reindex(columns=[r for r in ["North", "Central", "South"] if r in regions])
    fig = px.bar(tab, barmode="group", title="Negative reviews by topic and region (keyword classification)",
                 labels={"value": "reviews", "rule_aspect": ""}, color_discrete_sequence=["#8FA9CC", "#F4831F", "#1A3C6E"])
    fig.update_layout(height=360, legend_orientation="h", legend_y=-0.15, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, width="stretch")
    a = st.selectbox("Read reviews about", ["Delivery", "Packaging", "Service", "Taste", "Price", "Other"])
    st.dataframe(rv[rv["rule_aspect"] == a][["review_date", "channel", "region", "rating", "review_text", "rule_sentiment"]]
                 .sort_values("review_date", ascending=False).head(50), width="stretch", hide_index=True)
