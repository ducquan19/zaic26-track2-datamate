"""Few-shot generator: for each question, the most similar train questions with an SQL that ran to the gold answer and only
reads the canonical views, so it carries over to any shop. Retrieval is IDF-weighted overlap of words and character trigrams on
the question with accents dropped and digits masked, no extra model. Examples come from other shops only."""
import json
import math
import re
from collections import Counter
from pathlib import Path

from .ingest import norm

DEBT = {"debt_at", "debt_new", "debt_repaid"}
CANON = {"v_sales", "v_orders", "v_sales_all", "v_orders_all", "calendar"} | DEBT


def tables_of(sql):
    cte = {x.lower() for x in re.findall(r'(?i)(?:\bwith|,)\s*"?(\w+)"?\s+as\s*\(', sql)}
    return {x.lower() for x in re.findall(r'(?i)\b(?:from|join)\s+"?(\w+)"?', sql)} - cte


def feats(text):
    t = re.sub(r"\d+", "#", norm(text))
    words = [w for w in t.split("_") if w]
    return Counter(words) + Counter(t.replace("_", " ")[i:i + 3] for i in range(max(0, len(t) - 2)))


# Question shape: what is asked and over which period, read from the accent-free question. Two questions of one shape need
# the same SQL plan (ranking, period comparison, share, count at a grain, debt...) whatever the industry and item names.
SHAPES = [
    ("debt", r"(?:^|_)(?:no|cong_no|mua_chiu|ghi_no|tra_no|thu_no|con_no|tuoi_no)(?:_|$)"),
    ("lot", r"het_han|han_dung|han_su_dung|(?:^|_)hsd(?:_|$)|(?:^|_)lo_(?:hang|ton|thuoc|nao|gan|sap|con)(?:_|$)"),
    ("stock", r"(?:^|_)(?:ton|ton_kho|het_hang|con_hang|nhap_hang|nhap_ve|nha_cung_cap|ncc)(?:_|$)"),
    ("profit", r"(?:^|_)(?:lai|loi|loi_nhuan|gia_von|lai_gop|lai_rong)(?:_|$)"),
    ("status", r"(?:^|_)(?:huy|tra_hang|tra_lai|doi_tra|hoan|khong_giao_duoc|bi_tra)(?:_|$)"),
    ("rank", r"(?:^|_)(?:nhat|top|top_\d+|dan_dau|ban_chay|xep_hang|hang_dau)(?:_|$)"),
    ("compare", r"so_voi|(?:^|_)(?:tang|giam(?!_(?:gia|dau))|chenh_lech|hon_hay_kem|thay_doi|tang_truong|cao_hon|thap_hon)(?:_|$)"),
    ("share", r"(?:^|_)(?:ty_le|ti_le|phan_tram|chiem|ty_trong|ti_trong)(?:_|$)"),
    ("average", r"(?:^|_)(?:trung_binh|binh_quan|moi_don|moi_ngay|moi_khach|tb)(?:_|$)"),
    ("count", r"(?:^|_)(?:bao_nhieu|may|bn|bnhiu|so|dem)_(?:don|hoa_don|khach|ngay|lan|luot|mat_hang|san_pham|mon|lo|nguoi|bill)(?:_|$)"),
    ("list", r"liet_ke|(?:^|_)(?:nhung|cac)_(?:\w+_){0,3}nao(?:_|$)"),
    ("which_day", r"(?:^|_)(?:ngay|thang|hom|bua|tuan|khi)_nao(?:_|$)"),
    ("which_name", r"(?:^|_)(?:la_ai|la_gi|ai)(?:_|$)|(?:san_pham|mat_hang|mon|khach|nhan_vien|nhom|loai|thuoc)_(?:\w+_){0,2}nao(?:_|$)"),
    ("customer", r"(?:^|_)(?:khach|kh|khach_hang)(?:_|$)"),
    ("staff", r"(?:^|_)(?:nhan_vien|nv|tho|ca_lam|ca)(?:_|$)"),
    ("quantity", r"(?:^|_)(?:so_luong|sl|bao_nhieu_(?:cai|hop|chai|ly|goi|bao|tan|kg|vien))(?:_|$)"),
    ("money", r"(?:^|_)(?:doanh_thu|dthu|doanh_so|tien|thu_ve|thu_duoc|revenue)(?:_|$)"),
    ("p_day", r"(?:^|_)(?:hom_qua|hom_nay|hqua|hnay|bua_qua|bua_nay|ngay_\d|\d{1,2}_\d{1,2}_\d{4})(?:_|$)"),
    ("p_week", r"(?:^|_)tuan(?:_|$)"),
    ("p_month", r"(?:^|_)(?:thang|thg|th)(?:_|$)|(?:^|_)t\d{1,2}(?:_|$)"),
    ("p_quarter", r"(?:^|_)quy(?:_|$)"),
    ("p_ndays", r"(?:^|_)\d{1,3}_ngay"),
    ("p_tet", r"(?:^|_)tet(?:_|$)"),
]
SHAPES = [(k, re.compile(p)) for k, p in SHAPES]


def shape(question):
    """Set of shape tags of one question (SHAPES), e.g. {'rank', 'money', 'p_month'}."""
    t = norm(question)
    tags = {k for k, p in SHAPES if p.search(t)}
    if "%" in question:
        tags.add("share")
    return tags


def skeleton(sql):
    return re.sub(r"'[^']*'|\d+", "?", re.sub(r"\s+", " ", sql)).lower()


class Bank:
    def __init__(self, path):
        self.rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
        self.f = [feats(r["question"]) for r in self.rows]
        df = Counter(k for f in self.f for k in f)
        n = len(self.rows)
        self.idf = {k: math.log(1 + n / v) for k, v in df.items()}
        self.norm = [math.sqrt(sum((c * self.idf[k]) ** 2 for k, c in f.items())) for f in self.f]
        self.shapes = [shape(r["question"]) for r in self.rows]

    def retrieve(self, question, business, k=4, debt=False, by="text", exclude=()):
        """Top k rows, other shops only (and none of `exclude`), no two with the same SQL shape. Debt examples only where the shop has the debt tables.
        by: 'text' = cosine of words and trigrams (the submitted setup); 'shape' = overlap (Jaccard) of the question shape tags, cosine
        only breaking ties; 'hybrid' = the mean of both."""
        f = feats(question)
        nq = math.sqrt(sum((c * self.idf.get(x, 1.0)) ** 2 for x, c in f.items())) or 1.0
        qs = shape(question) if by != "text" else set()
        scored = []
        for i, g in enumerate(self.f):
            if self.rows[i]["business_id"] == business or self.rows[i]["business_id"] in exclude or (not debt and tables_of(self.rows[i]["sql"]) & DEBT):
                continue
            s = sum(c * g[x] * self.idf[x] ** 2 for x, c in f.items() if x in g) / (nq * self.norm[i] or 1.0)
            if by != "text":
                rs = self.shapes[i]
                j = len(qs & rs) / len(qs | rs) if qs | rs else 0.0
                s = j + 0.01 * s if by == "shape" else 0.5 * j + 0.5 * s
            scored.append((s, i))
        out, seen = [], set()
        for s, i in sorted(scored, reverse=True):
            sk = skeleton(self.rows[i]["sql"])
            if sk not in seen:
                seen.add(sk)
                out.append(self.rows[i])
            if len(out) == k:
                break
        return out


def block(rows):
    """The examples for the user message, kept right before the question."""
    if not rows:
        return ""
    text = "EXAMPLES FROM OTHER SHOPS (they show how a question maps to TYPE and SQL on the canonical views; this shop's RULEBOOK, SCHEMA and values decide, never copy a value or date from them):"
    for r in rows:
        text += f"\nQ: {r['question']}\nTYPE: {r['type']}\n```sql\n{r['sql']}\n```"
    return text
