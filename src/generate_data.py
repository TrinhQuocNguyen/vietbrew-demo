"""
VietBrew (fictional) — synthetic dataset generator for the upGrad x GGU MBA Bilingual demo class
"Advanced Data Analytics with Python".

VietBrew is a FICTIONAL direct-to-consumer coffee brand selling on Shopee, TikTok Shop, Lazada and its
own website. All numbers are SIMULATED. Patterns are modelled on public knowledge of the Vietnamese
market (Lunar New Year seasonality, double-day mega sales, cash-on-delivery, the 2025 province merger)
so that the class can discover them with Python.

Planted insights (what a good analysis should find):
  1. Seasonality follows the LUNAR calendar: the gift-box peak moves every year with Tết.
  2. TikTok Shop is the fastest-growing channel and has the best gross ROAS, but the worst net return
     after discounts, COD failures, returns, creator commissions and COGS.
  3. A small group of loyal customers produces a large share of revenue (RFM / K-Means finds 4 segments).
  4. Negative reviews cluster on DELIVERY in the CENTRAL region during the Tết peak, and on
     PACKAGING (dented gift boxes) — operations problems, not product problems.
  5. Province names change on 1 July 2025 (63 -> 34 provinces): naive charts show "collapsing" provinces.

Run:  python src/generate_data.py            (writes ./data, ~1.2M sales lines, takes ~1-2 minutes)
"""
from __future__ import annotations

import os
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 2026
rng = np.random.default_rng(SEED)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
(DATA / "raw").mkdir(parents=True, exist_ok=True)
(DATA / "instructor").mkdir(parents=True, exist_ok=True)

START, END = pd.Timestamp("2023-01-01"), pd.Timestamp("2026-11-30")
DAYS = pd.date_range(START, END, freq="D")
ND = len(DAYS)
D0 = np.datetime64("2023-01-01", "D")

# Tết Nguyên Đán (day 1 of lunar month 1, Vietnam)
TET = pd.to_datetime(["2023-01-22", "2024-02-10", "2025-01-29", "2026-02-17", "2027-02-06"])
MERGER_DATE = pd.Timestamp("2025-07-01")  # 34 provincial units operate from this date

CHANNELS = ["Shopee", "TikTok Shop", "Lazada", "Website"]

# --------------------------------------------------------------------------------------------
# Products (prices in VND, 2023 list price; +5% from 2025, +4% from 2026)
# --------------------------------------------------------------------------------------------
PRODUCTS = pd.DataFrame([
    # sku,        product_name (VN),                                 product_name_en,                         category,          price,   cost_ratio
    ("VB-RB500", "Cà phê hạt Robusta Đắk Lắk 500g",                   "Robusta beans, Dak Lak, 500 g",          "Beans & Ground",  145000, 0.42),
    ("VB-AR250", "Cà phê hạt Arabica Cầu Đất 250g",                   "Arabica beans, Cau Dat, 250 g",          "Beans & Ground",  165000, 0.45),
    ("VB-TT500", "Cà phê bột Truyền Thống 500g",                      "Traditional ground coffee, 500 g",       "Beans & Ground",  120000, 0.40),
    ("VB-CL250", "Cà phê bột Culi Đặc Biệt 250g",                     "Peaberry ground coffee, 250 g",          "Beans & Ground",   95000, 0.40),
    ("VB-DB10",  "Drip bag Arabica (hộp 10 gói)",                     "Arabica drip bags, box of 10",           "Drip & Instant",   99000, 0.38),
    ("VB-3IN1",  "Cà phê sữa hòa tan 3in1 (hộp 20 gói)",              "3-in-1 instant milk coffee, box of 20",  "Drip & Instant",   85000, 0.50),
    ("VB-CB6",   "Cold brew đóng chai 250ml (lốc 6)",                 "Bottled cold brew 250 ml, pack of 6",    "Cold Brew",       180000, 0.48),
    ("VB-PHIN",  "Phin nhôm truyền thống",                            "Aluminium phin filter",                  "Brewing Tools",    45000, 0.35),
    ("VB-SET",   "Bộ phin & ly sứ Bát Tràng",                         "Phin and ceramic cup set",               "Brewing Tools",   220000, 0.45),
    ("VB-GPQ",   "Hộp quà Tết Phú Quý",                               "Tết gift box 'Phu Quy'",                 "Tết Gift Box",    450000, 0.40),
    ("VB-GAK",   "Hộp quà Tết An Khang",                              "Tết gift box 'An Khang'",                "Tết Gift Box",    690000, 0.40),
    ("VB-GTV",   "Hộp quà Tết Thịnh Vượng (doanh nghiệp)",            "Tết gift box 'Thinh Vuong' (corporate)", "Tết Gift Box",   1250000, 0.38),
], columns=["sku", "product_name", "product_name_en", "category", "list_price_2023", "cost_ratio"])
PRODUCTS["unit_cost"] = (PRODUCTS["list_price_2023"] * PRODUCTS["cost_ratio"]).round(-3).astype(int)
SKU = {s: i for i, s in enumerate(PRODUCTS["sku"])}
PRICE23 = PRODUCTS["list_price_2023"].to_numpy()
IS_GIFT = (PRODUCTS["category"] == "Tết Gift Box").to_numpy()


def price_factor(years: np.ndarray) -> np.ndarray:
    return np.where(years >= 2026, 1.05 * 1.04, np.where(years >= 2025, 1.05, 1.0))


# --------------------------------------------------------------------------------------------
# Provinces: 63 old units -> 34 new units (Resolution 202/2025/QH15, effective 1 July 2025)
# (old_province, new_province, weight of the old unit inside the new one)
# --------------------------------------------------------------------------------------------
MERGE = {
    "Hà Nội": [("Hà Nội", 1)], "Huế": [("Thừa Thiên Huế", 1)], "Lai Châu": [("Lai Châu", 1)],
    "Điện Biên": [("Điện Biên", 1)], "Sơn La": [("Sơn La", 1)], "Lạng Sơn": [("Lạng Sơn", 1)],
    "Quảng Ninh": [("Quảng Ninh", 1)], "Thanh Hóa": [("Thanh Hóa", 1)], "Nghệ An": [("Nghệ An", 1)],
    "Hà Tĩnh": [("Hà Tĩnh", 1)], "Cao Bằng": [("Cao Bằng", 1)],
    "Tuyên Quang": [("Tuyên Quang", .5), ("Hà Giang", .5)],
    "Lào Cai": [("Lào Cai", .5), ("Yên Bái", .5)],
    "Thái Nguyên": [("Thái Nguyên", .8), ("Bắc Kạn", .2)],
    "Phú Thọ": [("Phú Thọ", .4), ("Vĩnh Phúc", .4), ("Hòa Bình", .2)],
    "Bắc Ninh": [("Bắc Ninh", .55), ("Bắc Giang", .45)],
    "Hưng Yên": [("Hưng Yên", .45), ("Thái Bình", .55)],
    "Hải Phòng": [("Hải Phòng", .65), ("Hải Dương", .35)],
    "Ninh Bình": [("Ninh Bình", .3), ("Hà Nam", .25), ("Nam Định", .45)],
    "Quảng Trị": [("Quảng Trị", .45), ("Quảng Bình", .55)],
    "Đà Nẵng": [("Đà Nẵng", .7), ("Quảng Nam", .3)],
    "Quảng Ngãi": [("Quảng Ngãi", .7), ("Kon Tum", .3)],
    "Gia Lai": [("Gia Lai", .4), ("Bình Định", .6)],
    "Khánh Hòa": [("Khánh Hòa", .75), ("Ninh Thuận", .25)],
    "Lâm Đồng": [("Lâm Đồng", .55), ("Đắk Nông", .15), ("Bình Thuận", .3)],
    "Đắk Lắk": [("Đắk Lắk", .65), ("Phú Yên", .35)],
    "TP. Hồ Chí Minh": [("TP. Hồ Chí Minh", .76), ("Bình Dương", .17), ("Bà Rịa - Vũng Tàu", .07)],
    "Đồng Nai": [("Đồng Nai", .75), ("Bình Phước", .25)],
    "Tây Ninh": [("Tây Ninh", .4), ("Long An", .6)],
    "Cần Thơ": [("Cần Thơ", .6), ("Sóc Trăng", .25), ("Hậu Giang", .15)],
    "Vĩnh Long": [("Vĩnh Long", .35), ("Bến Tre", .35), ("Trà Vinh", .3)],
    "Đồng Tháp": [("Đồng Tháp", .45), ("Tiền Giang", .55)],
    "Cà Mau": [("Cà Mau", .55), ("Bạc Liêu", .45)],
    "An Giang": [("An Giang", .5), ("Kiên Giang", .5)],
}
REGION = {}
for p in ["Hà Nội", "Hải Phòng", "Quảng Ninh", "Bắc Ninh", "Hưng Yên", "Ninh Bình", "Phú Thọ", "Thái Nguyên",
          "Tuyên Quang", "Lào Cai", "Lạng Sơn", "Cao Bằng", "Sơn La", "Điện Biên", "Lai Châu"]:
    REGION[p] = "North"
for p in ["Thanh Hóa", "Nghệ An", "Hà Tĩnh", "Quảng Trị", "Huế", "Đà Nẵng", "Quảng Ngãi", "Gia Lai",
          "Khánh Hòa", "Đắk Lắk", "Lâm Đồng"]:
    REGION[p] = "Central"
for p in ["TP. Hồ Chí Minh", "Đồng Nai", "Tây Ninh", "Cần Thơ", "Vĩnh Long", "Đồng Tháp", "An Giang", "Cà Mau"]:
    REGION[p] = "South"
assert len(MERGE) == 34 and sum(len(v) for v in MERGE.values()) == 63 and set(REGION) == set(MERGE)

# share of VietBrew customers living in each NEW province (simulated)
CUST_W = {
    "TP. Hồ Chí Minh": .30, "Hà Nội": .22, "Đà Nẵng": .035, "Hải Phòng": .035, "Bắc Ninh": .03, "Đồng Nai": .035,
    "Cần Thơ": .025, "Khánh Hòa": .018, "Lâm Đồng": .02, "Đắk Lắk": .015, "Gia Lai": .015, "Quảng Ngãi": .01,
    "Huế": .012, "Quảng Trị": .01, "Nghệ An": .02, "Thanh Hóa": .02, "Hưng Yên": .018, "Ninh Bình": .02,
    "Phú Thọ": .02, "Thái Nguyên": .012, "Quảng Ninh": .015, "Tây Ninh": .02, "Vĩnh Long": .015, "Đồng Tháp": .015,
    "An Giang": .018, "Cà Mau": .01, "Hà Tĩnh": .008, "Lào Cai": .006, "Tuyên Quang": .006, "Lạng Sơn": .005,
    "Cao Bằng": .003, "Sơn La": .005, "Điện Biên": .003, "Lai Châu": .002,
}
NEW_PROV = list(MERGE)
NEW_W = np.array([CUST_W[p] for p in NEW_PROV]); NEW_W /= NEW_W.sum()


def strip_accents(s: str) -> str:
    s = s.replace("đ", "d").replace("Đ", "D")
    return "".join(ch for ch in unicodedata.normalize("NFD", s) if unicodedata.category(ch) != "Mn")


# how people actually type province names in checkout forms
SPECIAL_VARIANTS = {
    "TP. Hồ Chí Minh": ["TP. Hồ Chí Minh", "TP.HCM", "Hồ Chí Minh", "Ho Chi Minh City", "HCM", "Sài Gòn", "tp hcm"],
    "Hà Nội": ["Hà Nội", "Ha Noi", "HN", "TP. Hà Nội", "Hanoi", "hà nội"],
    "Đà Nẵng": ["Đà Nẵng", "Da Nang", "TP. Đà Nẵng", "đà nẵng"],
    "Hải Phòng": ["Hải Phòng", "Hai Phong", "TP. Hải Phòng"],
    "Cần Thơ": ["Cần Thơ", "Can Tho", "TP. Cần Thơ"],
    "Thừa Thiên Huế": ["Thừa Thiên Huế", "Thua Thien Hue", "TT Huế", "Huế"],
    "Huế": ["Huế", "TP. Huế", "Hue"],
    "Bà Rịa - Vũng Tàu": ["Bà Rịa - Vũng Tàu", "Bà Rịa Vũng Tàu", "Vũng Tàu", "BR-VT", "Ba Ria - Vung Tau"],
}


def variant(name: str) -> list[str]:
    if name in SPECIAL_VARIANTS:
        return SPECIAL_VARIANTS[name]
    return [name, name, name, strip_accents(name), f"Tỉnh {name}", name.lower()]


# --------------------------------------------------------------------------------------------
# Calendar helpers
# --------------------------------------------------------------------------------------------
def days_to_next_tet(dates: pd.DatetimeIndex) -> np.ndarray:
    d = dates.values.astype("datetime64[D]")
    t = TET.values.astype("datetime64[D]")
    idx = np.searchsorted(t, d, side="left")
    return (t[np.clip(idx, 0, len(t) - 1)] - d).astype(int)


def days_from_prev_tet(dates: pd.DatetimeIndex) -> np.ndarray:
    d = dates.values.astype("datetime64[D]")
    t = TET.values.astype("datetime64[D]")
    idx = np.searchsorted(t, d, side="right") - 1
    out = (d - t[np.clip(idx, 0, len(t) - 1)]).astype(int)
    return np.where(idx < 0, 999, out)


DTT = days_to_next_tet(DAYS)            # days until next Tết, per calendar day
DFT = days_from_prev_tet(DAYS)          # days since last Tết
HOLIDAY = (DTT <= 2) | (DFT <= 4)        # Tết holiday: carriers stop, very few orders

# double-day mega sales (1.1 ... 12.12) and payday (25th)
MEGA_W = np.zeros(ND)
for y in range(2023, 2027):
    for m in range(1, 13):
        d = pd.Timestamp(year=y, month=m, day=m)
        if START <= d <= END:
            w = {9: 1.5, 10: 1.8, 11: 3.0, 12: 2.6, 6: 1.0, 7: 1.0, 8: 1.0}.get(m, 0.6)
            MEGA_W[(d - START).days] = w
        p = pd.Timestamp(year=y, month=m, day=25)
        if START <= p <= END:
            MEGA_W[(p - START).days] = max(MEGA_W[(p - START).days], 0.35)
MEGA_W[HOLIDAY] = 0
IS_MEGA = MEGA_W >= 0.6


def year_frac(day_idx: np.ndarray) -> np.ndarray:
    """0 at 2023-01-01, ~3.9 at 2026-11-30"""
    return day_idx / 365.25


def growth_dates(n: int, k: float, lo: int = 0, hi: int = ND - 30) -> np.ndarray:
    """sample n day-indices in [lo, hi) with density ~ exp(k * t): later dates more likely (business growth)"""
    t = np.arange(lo, hi)
    w = np.exp(k * year_frac(t)) * np.where(HOLIDAY[lo:hi], 0.15, 1.0)
    return rng.choice(t, size=n, p=w / w.sum())


def shift_out_of_holiday(day_idx: np.ndarray) -> np.ndarray:
    d = day_idx.copy()
    for _ in range(8):
        h = (d < ND) & HOLIDAY[np.clip(d, 0, ND - 1)]
        if not h.any():
            break
        keep = rng.random(h.sum()) < 0.15          # a few people still order during Tết
        d[np.flatnonzero(h)[~keep]] += 1
        if keep.all():
            break
    return d


# --------------------------------------------------------------------------------------------
# Customers
# --------------------------------------------------------------------------------------------
N_LOYAL, N_DEAL, N_TRIER = 15_000, 62_000, 98_000
GIFTERS_NEW = {2023: 3_000, 2024: 3_800, 2025: 4_600, 2026: 5_600}

segments = (["Loyal"] * N_LOYAL + ["Deal"] * N_DEAL + ["Trier"] * N_TRIER
            + ["Gifter"] * sum(GIFTERS_NEW.values()))
NC = len(segments)
cust = pd.DataFrame({"segment": segments})
cust["customer_id"] = [f"KH{i:06d}" for i in rng.permutation(np.arange(1, NC + 1))]
pi = rng.choice(len(NEW_PROV), size=NC, p=NEW_W)
cust["new_province"] = np.array(NEW_PROV)[pi]
old = []
for p in cust["new_province"]:
    names, w = zip(*MERGE[p])
    w = np.array(w, float)
    old.append(names[rng.choice(len(names), p=w / w.sum())])
cust["old_province"] = old
cust["region"] = cust["new_province"].map(REGION)
cust["corporate"] = (cust["segment"] == "Gifter") & (rng.random(NC) < 0.05)
cust["keeps_old_address"] = rng.random(NC) < 0.12   # still types the pre-merger province after 07/2025

orders = []   # list of dicts of arrays (one per segment), concatenated later


def add_orders(cidx, day, seg):
    day = np.asarray(day)
    cidx = np.broadcast_to(np.asarray(cidx), day.shape) if np.ndim(cidx) == 0 or len(np.atleast_1d(cidx)) != len(day) else np.asarray(cidx)
    ok = (day >= 0) & (day < ND)
    orders.append(pd.DataFrame({"cidx": cidx[ok], "day": day[ok], "seg": seg}))


# ---- Loyal brewers: regular repeat buyers ----
loyal_idx = np.flatnonzero(cust["segment"].eq("Loyal").to_numpy())
pre = rng.random(len(loyal_idx)) < 0.32                       # already customers before 2023
join = np.where(pre, rng.integers(0, 25, len(loyal_idx)), growth_dates(len(loyal_idx), 0.25))
life = rng.exponential(900, len(loyal_idx)).astype(int) + 120
gap_mean = rng.uniform(18, 45, len(loyal_idx))
for c, j, L, g in zip(loyal_idx, join, life, gap_mean):
    n = int((min(j + L, ND) - j) / g) + 3
    t = j + np.cumsum(rng.gamma(4, g / 4, n)).astype(int) - int(rng.uniform(0, g))
    t = t[(t >= max(j, 0)) & (t < min(j + L, ND))]
    if len(t) == 0:
        t = np.array([max(j, 0)])
    add_orders(c, shift_out_of_holiday(t), "Loyal")

# ---- Deal hunters: buy on double-day sales, mostly TikTok Shop ----
deal_idx = np.flatnonzero(cust["segment"].eq("Deal").to_numpy())
join = growth_dates(len(deal_idx), 0.55)
life = rng.exponential(520, len(deal_idx)).astype(int) + 30
mega_days = np.flatnonzero(MEGA_W > 0)
for c, j, L in zip(deal_idx, join, life):
    end = min(j + L, ND)
    n = 1 + rng.poisson(7.0 * (end - j) / 365.25)
    md = mega_days[(mega_days >= j) & (mega_days < end)]
    pick_mega = rng.random(n) < 0.78
    t = np.empty(n, int)
    if len(md):
        w = MEGA_W[md]
        t[pick_mega] = rng.choice(md, size=pick_mega.sum(), p=w / w.sum())
    else:
        pick_mega[:] = False
    t[~pick_mega] = rng.integers(j, max(end, j + 1), (~pick_mega).sum())
    t[0] = j
    add_orders(c, shift_out_of_holiday(np.unique(t)), "Deal")

# ---- One-time triers ----
trier_idx = np.flatnonzero(cust["segment"].eq("Trier").to_numpy())
join = growth_dates(len(trier_idx), 0.45)
# some triers arrive through mega sales
on_mega = rng.random(len(trier_idx)) < 0.25
w = MEGA_W[mega_days]; join[on_mega] = rng.choice(mega_days, on_mega.sum(), p=w / w.sum())
second = rng.random(len(trier_idx)) < 0.12
add_orders(trier_idx, shift_out_of_holiday(join), "Trier")
add_orders(trier_idx[second], shift_out_of_holiday(join[second] + rng.integers(20, 120, second.sum())), "Trier")

# ---- Tết gifters: one order per Tết season, repeat 50-70% ----
gift_idx = np.flatnonzero(cust["segment"].eq("Gifter").to_numpy())
pos = 0
season_of = {}
for y, n in GIFTERS_NEW.items():
    for c in gift_idx[pos:pos + n]:
        season_of[c] = y
    pos += n
tet_by_year = {y: TET[y - 2023] for y in range(2023, 2027)}
for c, first in season_of.items():
    corp = cust.at[c, "corporate"]
    for y in range(first, 2027):
        if y > first and rng.random() > (0.70 if corp else 0.50):
            break
        k = 1 + (rng.random() < 0.10)
        db = (6 + rng.gamma(3, 6, k)) if corp else (2 + rng.gamma(2.2, 5, k))
        db = np.clip(db, 3, 58).astype(int)
        d = ((tet_by_year[y] - START).days - db)
        add_orders(c, d, "Gifter")

O = pd.concat(orders, ignore_index=True)
O = O.sort_values(["day", "cidx"]).reset_index(drop=True)
NO = len(O)
print(f"orders: {NO:,}")

# --------------------------------------------------------------------------------------------
# Order attributes
# --------------------------------------------------------------------------------------------
yf = year_frac(O["day"].to_numpy())                          # 0..3.9
year = DAYS[O["day"]].year.to_numpy()
seg = O["seg"].to_numpy()
corp = cust["corporate"].to_numpy()[O["cidx"]]
region = cust["region"].to_numpy()[O["cidx"]]
dtt = DTT[O["day"]]
mega = IS_MEGA[O["day"]]


def mix(a, b, f):
    return np.array(a) * (1 - f) + np.array(b) * f


def pick_channel(seg_name, f):
    # probabilities over CHANNELS = [Shopee, TikTok Shop, Lazada, Website], drifting from 2023 (a) to 2026 (b)
    table = {
        "Loyal":  ([.44, .05, .13, .38], [.40, .14, .06, .40]),
        "Gifter": ([.34, .10, .12, .44], [.32, .26, .05, .37]),
        "Deal":   ([.45, .38, .15, .02], [.24, .70, .04, .02]),
        "Trier":  ([.47, .28, .15, .10], [.30, .60, .04, .06]),
    }
    a, b = table[seg_name]
    return mix(a, b, np.clip(f / 3.9, 0, 1)[:, None])


chan = np.empty(NO, dtype=object)
for s in ["Loyal", "Gifter", "Deal", "Trier"]:
    m = seg == s
    P = pick_channel(s, yf[m])
    u = rng.random(m.sum())[:, None]
    chan[m] = np.array(CHANNELS)[(u > np.cumsum(P, axis=1)).sum(axis=1).clip(0, 3)]
# deal-day orders lean to TikTok Shop livestreams
flip = (seg != "Gifter") & mega & (rng.random(NO) < 0.20)
chan[flip] = "TikTok Shop"

# payment method
PAY_PREPAID = {
    "Shopee": (["ShopeePay", "Credit/Debit card", "VNPay-QR"], [.65, .2, .15]),
    "TikTok Shop": (["MoMo", "ZaloPay", "Credit/Debit card"], [.45, .30, .25]),
    "Lazada": (["MoMo", "Credit/Debit card", "ZaloPay"], [.45, .35, .20]),
    "Website": (["Bank transfer", "MoMo", "VNPay-QR", "ZaloPay", "Credit/Debit card"], [.30, .25, .20, .10, .15]),
}
COD_SHARE = {"Shopee": (.58, .40), "TikTok Shop": (.80, .68), "Lazada": (.55, .42), "Website": (.32, .20)}
cod_p = np.array([COD_SHARE[c][0] for c in chan]) * (1 - yf / 3.9) + np.array([COD_SHARE[c][1] for c in chan]) * (yf / 3.9)
cod_p = np.where(seg == "Deal", np.minimum(cod_p + 0.08, 0.95), cod_p)
cod_p = np.where(corp, 0.05, cod_p)
is_cod = rng.random(NO) < cod_p
pay = np.empty(NO, dtype=object)
pay[is_cod] = "COD"
for c, (opts, w) in PAY_PREPAID.items():
    m = (~is_cod) & (chan == c)
    pay[m] = rng.choice(opts, size=m.sum(), p=w)
pay[(~is_cod) & corp] = "Bank transfer"

# order status
p_cancel = np.where(chan == "TikTok Shop", 0.05, 0.03)
p_fail = np.where(is_cod, pd.Series(chan).map({"TikTok Shop": .13, "Shopee": .06, "Lazada": .075, "Website": .03}).to_numpy(), 0.006)
p_ret = pd.Series(chan).map({"TikTok Shop": .07, "Shopee": .035, "Lazada": .042, "Website": .015}).to_numpy().astype(float)
p_ret = np.where(seg == "Deal", p_ret * 1.3, p_ret)
peak_central = (region == "Central") & (dtt >= 3) & (dtt <= 28)
p_fail = np.where(peak_central, p_fail * 2.0 + 0.02, p_fail)
u = rng.random(NO)
status = np.full(NO, "Delivered", dtype=object)
status[u < p_cancel + p_fail + p_ret] = "Returned"
status[u < p_cancel + p_fail] = "Failed delivery"
status[u < p_cancel] = "Cancelled"

# order time (hour): evenings; TikTok livestream peak 20-22h
hours = np.arange(24)
w_norm = np.array([1, .5, .3, .2, .2, .3, .8, 1.5, 2.2, 2.5, 2.6, 2.8, 3.4, 3.0, 2.4, 2.3, 2.4, 2.6, 2.8, 3.4, 4.2, 4.6, 3.6, 2.0])
w_tt = w_norm.copy(); w_tt[19:23] *= [1.6, 2.6, 2.8, 1.8]
hour = np.where(chan == "TikTok Shop", rng.choice(hours, NO, p=w_tt / w_tt.sum()), rng.choice(hours, NO, p=w_norm / w_norm.sum()))
minute = rng.integers(0, 60, NO); second = rng.integers(0, 60, NO)

# order ids
O["order_id"] = [f"VB{y % 100:02d}{i:07d}" for i, y in enumerate(year, start=1)]

# --------------------------------------------------------------------------------------------
# Order lines
# --------------------------------------------------------------------------------------------
MIX = {
    "Loyal": {"VB-RB500": .25, "VB-AR250": .20, "VB-TT500": .14, "VB-CL250": .10, "VB-DB10": .15, "VB-CB6": .09, "VB-3IN1": .03, "VB-PHIN": .02, "VB-SET": .02},
    "Deal":  {"VB-DB10": .30, "VB-3IN1": .30, "VB-TT500": .15, "VB-CB6": .10, "VB-CL250": .10, "VB-PHIN": .05},
    "Trier": {"VB-DB10": .25, "VB-3IN1": .20, "VB-TT500": .15, "VB-RB500": .15, "VB-CB6": .10, "VB-SET": .05, "VB-PHIN": .05, "VB-AR250": .05},
}
MIX_ARR = {k: (np.array([SKU[s] for s in v]), np.array(list(v.values())) / sum(v.values())) for k, v in MIX.items()}

line_o, line_p, line_q = [], [], []
nl = np.ones(NO, int)
nl[seg == "Loyal"] = 1 + rng.binomial(2, 0.62, (seg == "Loyal").sum())
nl[seg == "Deal"] = 1 + rng.binomial(2, 0.40, (seg == "Deal").sum())
nl[seg == "Trier"] = 1 + rng.binomial(2, 0.25, (seg == "Trier").sum())
for s in ["Loyal", "Deal", "Trier"]:
    idx = np.flatnonzero(seg == s)
    skus, w = MIX_ARR[s]
    for k in range(1, 4):
        sub = idx[nl[idx] == k]
        if not len(sub):
            continue
        # sample k distinct products per order (Gumbel top-k trick)
        g = np.log(w)[None, :] - np.log(-np.log(rng.random((len(sub), len(w)))))
        top = np.argsort(-g, axis=1)[:, :k]
        line_o.append(np.repeat(sub, k)); line_p.append(skus[top].ravel())
        qmax = 4 if s == "Deal" else 3
        line_q.append(1 + rng.binomial(qmax - 1, 0.18 if s != "Deal" else 0.30, len(sub) * k))

# gift boxes for loyal / trier orders placed in the Tết window
for s, p_add in [("Loyal", 0.28), ("Trier", 0.40)]:
    idx = np.flatnonzero((seg == s) & (dtt >= 4) & (dtt <= 45) & (rng.random(NO) < p_add))
    line_o.append(idx); line_p.append(rng.choice([SKU["VB-GPQ"], SKU["VB-GAK"]], len(idx), p=[.6, .4]))
    line_q.append(1 + rng.binomial(3, 0.25, len(idx)))

# gifters
gidx = np.flatnonzero(seg == "Gifter")
for o in gidx:
    if corp[o]:
        boxes = int(rng.integers(10, 40))
        types = rng.choice([SKU["VB-GTV"], SKU["VB-GAK"]], size=1 + (rng.random() < 0.35), replace=False, p=[.55, .45])
    else:
        boxes = int(1 + rng.binomial(6, 0.20))
        types = rng.choice([SKU["VB-GPQ"], SKU["VB-GAK"], SKU["VB-GTV"]], size=1 + (rng.random() < 0.18 and boxes > 1), replace=False, p=[.55, .35, .10])
    split = np.array([boxes]) if len(types) == 1 else np.array([boxes - boxes // 2, boxes // 2])
    for t, q in zip(types, split):
        line_o.append([o]); line_p.append([t]); line_q.append([max(1, q)])
    if rng.random() < 0.18:
        line_o.append([o]); line_p.append([SKU["VB-RB500"] if rng.random() < .6 else SKU["VB-AR250"]]); line_q.append([1])

L = pd.DataFrame({"o": np.concatenate(line_o), "p": np.concatenate(line_p), "qty": np.concatenate(line_q)}).astype(int)
L = L.sort_values(["o", "p"]).drop_duplicates(["o", "p"]).reset_index(drop=True)
print(f"lines: {len(L):,}")

o = L["o"].to_numpy(); p = L["p"].to_numpy(); q = L["qty"].to_numpy()
ly = year[o]
unit_price = (np.round(PRICE23[p] * price_factor(ly) / 1000) * 1000).astype(int)

# discounts
base = pd.Series(chan[o]).map({"Website": .03, "Shopee": .08, "Lazada": .10, "TikTok Shop": .16}).to_numpy()
rate = base + np.where(mega[o], 0.12, 0) + np.where(seg[o] == "Deal", 0.08, 0) + rng.normal(0, 0.03, len(L))
rate = np.where(IS_GIFT[p] & (q >= 10), np.maximum(rate, 0.10), rate)       # volume discount for corporate
rate = np.where(IS_GIFT[p] & ~mega[o], rate * 0.6, rate)                     # gift boxes rarely discounted
rate = np.clip(rate, 0, 0.45)
discount = (np.round(rate * unit_price * q / 1000) * 1000).astype(int)

# --------------------------------------------------------------------------------------------
# Province text as typed (old names before 1 July 2025, new names after — with variants)
# --------------------------------------------------------------------------------------------
odate = DAYS[O["day"].to_numpy()]
after = odate >= MERGER_DATE
new_p = cust["new_province"].to_numpy()[O["cidx"]]
old_p = cust["old_province"].to_numpy()[O["cidx"]]
keeps_old = cust["keeps_old_address"].to_numpy()[O["cidx"]]
canon = np.where(after & ~keeps_old, new_p, old_p)
var_cache = {n: variant(n) for n in set(canon)}
pv = np.array([var_cache[n][rng.integers(0, len(var_cache[n]))] for n in canon], dtype=object)

# --------------------------------------------------------------------------------------------
# Assemble raw sales export (one row per order line)
# --------------------------------------------------------------------------------------------
ts = (odate.values + hour.astype("timedelta64[h]") + minute.astype("timedelta64[m]") + second.astype("timedelta64[s]"))


def vn_money(a: np.ndarray) -> np.ndarray:
    # Vietnamese thousands separator: 145000 -> "145.000"
    return np.array([f"{v:,}".replace(",", ".") for v in a], dtype=object)


raw = pd.DataFrame({
    "order_id": O["order_id"].to_numpy()[o],
    "order_datetime": pd.to_datetime(ts[o]).strftime("%Y-%m-%d %H:%M:%S"),
    "customer_id": cust["customer_id"].to_numpy()[O["cidx"].to_numpy()[o]],
    "channel": chan[o],
    "province": pv[o],
    "sku": PRODUCTS["sku"].to_numpy()[p],
    "product_name": PRODUCTS["product_name"].to_numpy()[p],
    "category": PRODUCTS["category"].to_numpy()[p],
    "quantity": q,
    "unit_price": vn_money(unit_price),
    "discount_vnd": discount,
    "payment_method": pay[o],
    "order_status": status[o],
})
# a realistic export glitch: ~0.3% duplicated rows
dup = raw.sample(frac=0.003, random_state=SEED)
raw = pd.concat([raw, dup]).sort_values(["order_datetime", "order_id", "sku"]).reset_index(drop=True)
print(f"raw rows (with duplicates): {len(raw):,}")

raw_path = DATA / "raw" / "vietbrew_sales_2023_2026.csv.gz"
raw.to_csv(raw_path, index=False, compression={"method": "gzip", "compresslevel": 6})
print("wrote", raw_path, f"{raw_path.stat().st_size / 1e6:.1f} MB")

# --------------------------------------------------------------------------------------------
# Reference tables
# --------------------------------------------------------------------------------------------
prod_out = PRODUCTS[["sku", "product_name", "product_name_en", "category", "list_price_2023", "unit_cost"]].copy()
prod_out["list_price_2026"] = (np.round(prod_out["list_price_2023"] * 1.05 * 1.04 / 1000) * 1000).astype(int)
prod_out.to_csv(DATA / "products.csv", index=False)

pd.DataFrame({"tet_year": range(2023, 2028), "tet_date": TET.strftime("%Y-%m-%d")}).to_csv(DATA / "tet_dates.csv", index=False)

rows = []
for newp, comps in MERGE.items():
    for oldp, _ in comps:
        rows.append({"old_province": oldp, "new_province": newp, "region": REGION[newp]})
pd.DataFrame(rows).to_csv(DATA / "province_mapping_2025.csv", index=False)

# channel cost assumptions (illustrative, for the net-contribution exercise)
pd.DataFrame([
    {"channel": "Shopee",      "platform_commission": 0.10, "affiliate_commission": 0.02, "payment_fee_prepaid": 0.00},
    {"channel": "TikTok Shop", "platform_commission": 0.08, "affiliate_commission": 0.12, "payment_fee_prepaid": 0.00},
    {"channel": "Lazada",      "platform_commission": 0.10, "affiliate_commission": 0.02, "payment_fee_prepaid": 0.00},
    {"channel": "Website",     "platform_commission": 0.00, "affiliate_commission": 0.00, "payment_fee_prepaid": 0.02},
]).assign(fulfilment_cost_per_order=15000, reverse_logistics_per_failed_or_returned=30000,
          returned_goods_write_off=0.5).to_csv(DATA / "channel_costs.csv", index=False)

# --------------------------------------------------------------------------------------------
# Marketing spend by month x channel (targets: gross ROAS TikTok > Shopee > Website > Lazada)
# --------------------------------------------------------------------------------------------
ROAS_TARGET = {"TikTok Shop": 11.0, "Shopee": 8.2, "Website": 6.2, "Lazada": 5.4}
gmv = raw.drop_duplicates()
gmv = gmv[gmv["order_status"] != "Cancelled"].copy()
gmv["gmv"] = gmv["quantity"] * gmv["unit_price"].str.replace(".", "", regex=False).astype(int) - gmv["discount_vnd"]
gmv["month"] = gmv["order_datetime"].str[:7]
mg = gmv.groupby(["month", "channel"], as_index=False)["gmv"].sum()
mg["ad_spend_vnd"] = (mg["gmv"] / mg["channel"].map(ROAS_TARGET) * rng.uniform(0.88, 1.12, len(mg)) / 1e5).round() * 1e5
mg[["month", "channel", "ad_spend_vnd"]].astype({"ad_spend_vnd": "int64"}).to_csv(DATA / "marketing_spend.csv", index=False)

# --------------------------------------------------------------------------------------------
# Customer reviews (Vietnamese, with teencode) — 3,000 reviews
# --------------------------------------------------------------------------------------------
T = {
    ("Delivery", "Positive"): [
        "Giao hàng nhanh, đặt hôm qua hôm nay đã nhận được", "Ship siêu nhanh, shipper thân thiện",
        "Đặt tối giao sáng, quá ổn", "Giao hàng đúng hẹn, kịp mang đi biếu Tết", "Giao nhanh hơn mình nghĩ",
        "Shipper gọi trước khi giao, rất chu đáo"],
    ("Delivery", "Negative"): [
        "Giao hàng quá chậm, đặt 1 tuần mới nhận", "Ship lâu quá trời, gần Tết mà chờ mãi không thấy",
        "Đơn hàng bị trễ 5 ngày, không ai báo gì", "Shipper hẹn mấy lần không tới, mua quà Tết mà về sau Tết",
        "Giao trễ, quà tới nơi thì người nhận đã về quê ăn Tết", "Chờ dài cổ mới có hàng, lần sau không đặt nữa",
        "Đơn kẹt ở kho trung chuyển cả tuần", "Hàng giao chậm, gọi tổng đài không ai nghe"],
    ("Packaging", "Positive"): [
        "Hộp quà đẹp sang, tặng đối tác rất ưng", "Đóng gói chắc chắn, có túi giấy kèm theo",
        "Bao bì xịn, thiết kế đẹp, đáng tiền", "Gói hàng cẩn thận, chống sốc kỹ"],
    ("Packaging", "Negative"): [
        "Hộp quà bị móp méo, không dám mang đi biếu", "Hộp quà Tết bị móp góc, túi bị rách",
        "Đóng gói sơ sài, cà phê bị đổ ra ngoài", "Hộp bị bể, ruy băng rơi mất, thất vọng",
        "Hộp quà nhận về bị trầy xước nhiều", "Thùng carton ướt, hộp quà bên trong bị biến dạng"],
    ("Taste", "Positive"): [
        "Cà phê thơm, vị đậm, uống đúng gu", "Robusta đậm đà, pha phin rất ngon", "Arabica chua nhẹ, hậu ngọt, sẽ mua lại",
        "Drip bag thơm, tiện mang đi làm", "Cold brew mát lạnh, vị dịu, cả nhà đều thích",
        "Cà phê sữa hòa tan vừa ngọt, không bị gắt", "Mùi thơm lắm, mở túi ra là thơm cả phòng", "Uống quen rồi, lần nào cũng ngon"],
    ("Taste", "Negative"): [
        "Cà phê bị chua, không hợp khẩu vị", "Vị nhạt, không giống mô tả", "Cà phê hơi khét, đắng gắt",
        "Lô này không thơm như lô trước"],
    ("Price", "Positive"): ["Giá hợp lý, săn sale được giá rất tốt", "Giá ok so với chất lượng", "Mua đợt 11.11 được giảm sâu, quá hời"],
    ("Price", "Negative"): ["Giá hơi cao so với mặt bằng chung", "Không sale thì hơi đắt"],
    ("Service", "Positive"): ["Shop tư vấn nhiệt tình, trả lời tin nhắn nhanh", "Shop hỗ trợ đổi hàng nhanh gọn"],
    ("Service", "Negative"): ["Giao sai sản phẩm, nhắn shop mãi không trả lời", "Đặt hộp An Khang mà giao hộp Phú Quý",
                               "Thiếu hàng trong đơn, shop hẹn bù mà không thấy"],
    ("Other", "Neutral"): ["Hàng ổn, đúng mô tả", "Tạm được, để uống thử thêm", "Bình thường, không có gì đặc biệt"],
}
EXTRA_POS = ["sẽ ủng hộ shop dài dài", "lần thứ 3 mua rồi", "đóng gói cũng đẹp", "giá cũng ok", "cho shop 5 sao"]
EXTRA_NEG = ["rất thất vọng", "mong shop xem lại", "không đáng tiền", "lần đầu mua đã gặp chuyện này"]
EMO_POS = ["😍", "👍", "❤️", "☕", "🥰", "", "", ""]
EMO_NEG = ["😡", "😤", "👎", "😢", "", "", ""]
TEEN = [("không", ["k", "ko", "hông", "hok"]), ("được", ["dc", "đc", "đk"]), ("quá", ["wá", "qá"]), ("rồi", ["r", "rùi"]),
        ("cà phê", ["cf", "cafe", "cà fê"]), ("mình", ["mk", "mik"]), ("giao hàng", ["ship", "giao"]), ("nhanh", ["nhanhhh", "nhanh"]),
        ("thơm", ["thơmmm", "thơm"]), ("ngon", ["ngonnn", "ngon"]), ("vậy", ["z", "v"])]


def teencode(s: str) -> str:
    for a, opts in TEEN:
        if a in s and rng.random() < 0.6:
            s = s.replace(a, opts[rng.integers(0, len(opts))])
    return s


lines_ok = L.assign(status=status[o], gift=IS_GIFT[p])
cand = lines_ok[lines_ok["status"].isin(["Delivered", "Returned"])].groupby("o").agg(gift=("gift", "max"), status=("status", "first")).reset_index()
cand["region"] = region[cand["o"]]
cand["dtt"] = dtt[cand["o"]]
cand["peak"] = (cand["dtt"] >= 3) & (cand["dtt"] <= 28)
# who writes reviews: unhappy customers write more often
wt = np.where(cand["status"].eq("Returned"), 2.5, 1.0)
wt = np.where(cand["peak"] & cand["region"].eq("Central"), wt * 3.0, wt)
wt = np.where(cand["gift"], wt * 2.0, wt)
R = cand.sample(n=3000, weights=wt, random_state=SEED).reset_index(drop=True)

aspects, sents, texts = [], [], []
for _, r in R.iterrows():
    u = rng.random()
    if r["status"] == "Returned":
        a, s = (("Packaging", "Negative") if (r["gift"] and u < .65) or u < .30 else ("Taste", "Negative") if u < .48 else ("Service", "Negative"))
    elif r["peak"] and r["region"] == "Central" and u < .50:
        a, s = "Delivery", "Negative"
    elif r["peak"] and u < .07:
        a, s = "Delivery", "Negative"
    elif r["gift"] and u < .20:
        a, s = "Packaging", "Negative"
    else:
        a, s = [("Taste", "Positive"), ("Delivery", "Positive"), ("Packaging", "Positive"), ("Price", "Positive"),
                ("Other", "Neutral"), ("Taste", "Negative"), ("Price", "Negative"), ("Service", "Positive"), ("Delivery", "Negative")][
            rng.choice(9, p=[.43, .17, .12, .08, .08, .025, .025, .04, .03])]
    t = T[(a, s)][rng.integers(0, len(T[(a, s)]))]
    if s == "Positive" and rng.random() < .45:
        t += ", " + EXTRA_POS[rng.integers(0, len(EXTRA_POS))]
    if s == "Negative" and rng.random() < .45:
        t += ", " + EXTRA_NEG[rng.integers(0, len(EXTRA_NEG))]
    if rng.random() < .5:
        t = teencode(t.lower())
    if rng.random() < .15:
        t = strip_accents(t)
    e = (EMO_POS if s == "Positive" else EMO_NEG if s == "Negative" else [""])
    t = (t + " " + e[rng.integers(0, len(e))]).strip()
    aspects.append(a); sents.append(s); texts.append(t)

rating = np.where(np.array(sents) == "Positive", rng.choice([4, 5], len(R), p=[.3, .7]),
                  np.where(np.array(sents) == "Negative", rng.choice([1, 2, 3], len(R), p=[.45, .4, .15]), 3))
rev_day = np.minimum(R["o"].map(lambda i: O.at[i, "day"]).to_numpy() + rng.integers(2, 12, len(R)), ND - 1)
oi = R["o"].to_numpy()
cat_main = lines_ok.groupby("o")["p"].first().reindex(oi).to_numpy()
reviews = pd.DataFrame({
    "review_id": [f"RV{i:05d}" for i in range(1, len(R) + 1)],
    "order_id": O["order_id"].to_numpy()[oi],
    "review_date": DAYS[rev_day].strftime("%Y-%m-%d"),
    "channel": chan[oi],
    "province": new_p[oi],
    "region": region[oi],
    "product_category": np.where(R["gift"], "Tết Gift Box", PRODUCTS["category"].to_numpy()[cat_main]),
    "rating": rating,
    "review_text": texts,
}).sort_values("review_date").reset_index(drop=True)
labels = pd.DataFrame({"review_id": reviews["review_id"]})
lab = pd.DataFrame({"aspect": aspects, "sentiment": sents}, index=[f"RV{i:05d}" for i in range(1, len(R) + 1)])
labels = labels.join(lab, on="review_id")
reviews.to_csv(DATA / "reviews.csv", index=False, encoding="utf-8-sig")
labels.to_csv(DATA / "instructor" / "reviews_labels.csv", index=False)

# customers ground truth (instructor only)
seen = cust.iloc[np.unique(O["cidx"])]
seen[["customer_id", "segment", "corporate", "new_province", "region"]].to_csv(DATA / "instructor" / "customers_truth.csv", index=False)

print("done:", {k: len(v) for k, v in {"raw": raw, "reviews": reviews, "customers": seen}.items()})
