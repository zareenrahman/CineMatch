# =========================
# file: recsys/recsys_core.py
# =========================
from __future__ import annotations

import io
import json
import os
import re
import zipfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from scipy import sparse

# Download sources
ML100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"

# ---------- utils ----------
def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _download_zip(url: str, dest_dir: str) -> None:
    _ensure_dir(dest_dir)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        zf.extractall(dest_dir)  # why: avoids partial files

def _has_1m(data_dir: str) -> bool:
    return os.path.exists(os.path.join(data_dir, "ml-1m", "ratings.dat"))

def _has_100k(data_dir: str) -> bool:
    return os.path.exists(os.path.join(data_dir, "ml-100k", "u.data"))

def _bootstrap(data_dir: str) -> None:
    if _has_1m(data_dir) or _has_100k(data_dir):
        return
    try:
        _download_zip(ML1M_URL, data_dir)
        return
    except Exception:
        pass
    _download_zip(ML100K_URL, data_dir)

# ---------- loaders ----------
def _load_ml1m(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    base = os.path.join(data_dir, "ml-1m")
    ratings = pd.read_csv(
        os.path.join(base, "ratings.dat"),
        sep="::", engine="python", header=None,
        names=["userId", "movieId", "rating", "timestamp"],
    )
    movies = pd.read_csv(
        os.path.join(base, "movies.dat"),
        sep="::", engine="python", header=None, encoding="latin-1",
        names=["movieId", "title", "genres"],
    )
    return ratings, movies

def _load_ml100k(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    base = os.path.join(data_dir, "ml-100k")
    ratings = pd.read_csv(
        os.path.join(base, "u.data"),
        sep="\t", engine="python", header=None,
        names=["userId", "movieId", "rating", "timestamp"],
    )
    raw = pd.read_csv(
        os.path.join(base, "u.item"),
        sep="|", engine="python", header=None, encoding="latin-1",
    )
    raw = raw.rename(columns={0: "movieId", 1: "title"})
    if raw.shape[1] > 5:
        genre_cols = list(range(5, raw.shape[1]))
        genre_names = [
            "unknown","Action","Adventure","Animation","Children's","Comedy","Crime","Documentary","Drama",
            "Fantasy","Film-Noir","Horror","Musical","Mystery","Romance","Sci-Fi","Thriller","War","Western"
        ][:len(genre_cols)]
        gdf = raw[genre_cols]
        genres = gdf.apply(lambda r: "|".join([g for g, v in zip(genre_names, r) if int(v)==1]), axis=1)
    else:
        genres = ""
    movies = pd.DataFrame({"movieId": raw["movieId"], "title": raw["title"], "genres": genres})
    return ratings, movies

# ---------- normalization & stats ----------
_YEAR_RE = re.compile(r"\((\d{4})\)")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")

def clean_title(title: str) -> Tuple[str, Optional[int]]:
    if not isinstance(title, str):
        return "", None
    year = None
    m = _YEAR_RE.search(title)
    if m:
        try: year = int(m.group(1))
        except Exception: year = None
    base = _YEAR_RE.sub("", title).lower().strip()
    base = _NON_ALNUM.sub(" ", base)
    base = re.sub(r"\s+", " ", base).strip()
    return base, year

def add_title_features(movies: pd.DataFrame) -> pd.DataFrame:
    base, years = zip(*movies["title"].map(clean_title))
    out = movies.copy()
    out["clean_title"] = list(base)
    out["year"] = list(years)
    return out

def _stats(ratings: pd.DataFrame) -> pd.DataFrame:
    g = ratings.groupby("movieId")["rating"]
    return g.mean().to_frame("avg_rating").join(g.count().to_frame("n_ratings")).reset_index()

# ---------- item-item similarity (for “Similar titles”) ----------
def _build_maps(ratings: pd.DataFrame, movies: pd.DataFrame):
    uids = np.sort(ratings["userId"].unique()); iids = np.sort(movies["movieId"].unique())
    u_map = {u:i for i,u in enumerate(uids)}; i_map = {m:j for j,m in enumerate(iids)}
    return u_map, i_map, {i:u for u,i in u_map.items()}, {j:m for m,j in i_map.items()}

def _matrix(ratings: pd.DataFrame, u_map, i_map) -> sparse.csr_matrix:
    rows = ratings["userId"].map(u_map).values
    cols = ratings["movieId"].map(i_map).values
    vals = ratings["rating"].astype(np.float32).values
    return sparse.coo_matrix((vals,(rows,cols)), shape=(len(u_map),len(i_map))).tocsr().astype(np.float32)

def _binarize(R: sparse.csr_matrix, thr: float = 4.0) -> sparse.csr_matrix:
    data = (R.data >= thr).astype(np.float32)
    return sparse.csr_matrix((data, R.indices, R.indptr), shape=R.shape)

def _topk_item_cosine(R_bin: sparse.csr_matrix, k: int = 50) -> sparse.csr_matrix:
    n_items = R_bin.shape[1]; norms = np.sqrt(R_bin.multiply(R_bin).sum(axis=0)).A1 + 1e-8
    indptr=[0]; indices=[]; data=[]; block=1024; Rt=R_bin.T.tocsr()
    for start in range(0,n_items,block):
        stop=min(start+block,n_items)
        numer = Rt @ R_bin[:,start:stop]
        denom = norms[:,None]*norms[start:stop][None,:]
        cs = numer.multiply(sparse.csr_matrix(1.0/denom)); cs.setdiag(0)
        for j in range(cs.shape[1]):
            col = cs.getcol(j).tocoo()
            if col.nnz==0: indptr.append(indptr[-1]); continue
            if col.nnz>k:
                top = np.argpartition(-col.data,k-1)[:k]; idx=col.row[top]; val=col.data[top]
                order=np.argsort(-val); idx=idx[order]; val=val[order]
            else:
                idx=col.row; val=col.data; order=np.argsort(-val); idx=idx[order]; val=val[order]
            indices.extend(idx.tolist()); data.extend(val.tolist()); indptr.append(indptr[-1]+len(idx))
    return sparse.csr_matrix((np.array(data,np.float32),np.array(indices,np.int32),np.array(indptr,np.int32)), shape=(n_items,n_items))

def _save_sparse(path: str, mat: sparse.csr_matrix, meta: dict) -> None:
    _ensure_dir(os.path.dirname(path))
    np.savez_compressed(path, data=mat.data, indices=mat.indices, indptr=mat.indptr, shape=mat.shape, meta=json.dumps(meta))

def _load_sparse(path: str) -> Tuple[sparse.csr_matrix, dict]:
    z = np.load(path, allow_pickle=False)
    return sparse.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"])), json.loads(str(z["meta"]))

# ---------- CF data bundle ----------
@dataclass
class DataBundle:
    data_dir: str
    ratings: pd.DataFrame
    movies: pd.DataFrame
    user_map: Dict[int,int]
    item_map: Dict[int,int]
    inv_user_map: Dict[int,int]
    inv_item_map: Dict[int,int]
    R: sparse.csr_matrix
    item_sims: sparse.csr_matrix
    item_stats: pd.DataFrame

def bootstrap_data(data_dir: str = "data") -> DataBundle:
    _ensure_dir(data_dir)
    try:
        _bootstrap(data_dir)
    except Exception as e:
        raise RuntimeError(f"Bootstrap failed: {e}")
    if _has_1m(data_dir):
        ratings, movies = _load_ml1m(data_dir)
    else:
        ratings, movies = _load_ml100k(data_dir)
    movies = add_title_features(movies)
    user_map, item_map, inv_user_map, inv_item_map = _build_maps(ratings, movies)
    ratings = ratings[ratings["movieId"].isin(item_map)].copy()
    R = _matrix(ratings, user_map, item_map)
    cache = os.path.join(data_dir, "item_sims_topk.npz")
    sims = None
    if os.path.exists(cache):
        try:
            sims, meta = _load_sparse(cache)
            if meta.get("shape") != list(R.shape): sims = None
        except Exception:
            sims = None
    if sims is None:
        sims = _topk_item_cosine(_binarize(R,4.0), k=50)
        _save_sparse(cache, sims, {"shape": list(R.shape)})
    return DataBundle(
        data_dir=data_dir, ratings=ratings, movies=movies,
        user_map=user_map, item_map=item_map, inv_user_map=inv_user_map, inv_item_map=inv_item_map,
        R=R, item_sims=sims, item_stats=_stats(ratings)
    )

# ---------- search + item info ----------


_YEAR_NUMBER = re.compile(r"\b(\d{4})\b")  # local helper; doesn't shadow the earlier one

def search_movies(bundle, query: str, limit: int = 20) -> pd.DataFrame:
    """
    Case-insensitive, year-agnostic substring search.
    Accepts 'toy story' or 'Toy Story (1995)' etc.
    Safe against regex chars in the query.
    """
    q_raw = (query or "").strip()
    if not q_raw:
        return pd.DataFrame(columns=["movieId", "title", "genres", "year"])

    # Optional year anywhere in the input
    m = _YEAR_NUMBER.search(q_raw)
    q_year = int(m.group(1)) if m else None

    # Remove year marker and trim the text part
    q = _YEAR_NUMBER.sub("", q_raw).strip()

    df = bundle.movies.copy()

    # Match in title OR clean_title; `case=False` makes it case-insensitive,
    # `regex=False` prevents special-character issues in the query.
    mask = (
        df["title"].str.contains(q, case=False, na=False, regex=False) |
        df["clean_title"].str.contains(q, case=False, na=False, regex=False)
    )
    out = df[mask].copy()

    # Bias to same year if provided
    if q_year:
        out["_year_bias"] = (out["year"] == q_year).astype(int)
        out = out.sort_values(["_year_bias", "title"], ascending=[False, True]).drop(columns=["_year_bias"])
    else:
        out = out.sort_values("title")

    return out.head(limit)[["movieId", "title", "genres", "year"]]
    
def similar_items(bundle: DataBundle, movie_id: int, k: int = 10) -> pd.DataFrame:
    if movie_id not in bundle.item_map: return pd.DataFrame(columns=["movieId","title","genres","year","similarity"])
    j = bundle.item_map[movie_id]; col = bundle.item_sims.getrow(j)
    if col.nnz == 0: return pd.DataFrame(columns=["movieId","title","genres","year","similarity"])
    idx = col.indices[:k]; val = col.data[:k]
    mids = [bundle.inv_item_map[i] for i in idx]
    info = bundle.movies.set_index("movieId").loc[mids][["title","genres","year"]].reset_index()
    sim = (val - val.min())/(val.max()-val.min()+1e-8)
    info["similarity"] = sim
    return info

def movie_card(bundle: DataBundle, movie_id: int) -> Dict[str, object]:
    row = bundle.movies.set_index("movieId").loc[movie_id]
    st_row = bundle.item_stats.set_index("movieId").reindex([movie_id])
    avg = float(st_row["avg_rating"].iloc[0]) if not st_row.isna().any().any() else np.nan
    cnt = int(st_row["n_ratings"].iloc[0]) if not st_row.isna().any().any() else 0
    return {
        "movieId": int(movie_id),
        "title": str(row["title"]),
        "year": int(row["year"]) if pd.notna(row["year"]) else None,
        "genres": str(row.get("genres","") or ""),
        "avg_rating": None if np.isnan(avg) else round(avg,2),
        "n_ratings": cnt,
        "available": True,
    }

# ---------- YOUR kNN (user-based Pearson) ----------
from collections import defaultdict
import math

@dataclass
class UIData:
    by_user: Dict[int, Dict[int, float]]
    by_item: Dict[int, Dict[int, float]]
    user_means: Dict[int, float]
    users: set[int]
    items: set[int]

def build_ui_data(ratings: pd.DataFrame) -> UIData:
    by_user: Dict[int, Dict[int, float]] = defaultdict(dict)
    by_item: Dict[int, Dict[int, float]] = defaultdict(dict)
    for row in ratings.itertuples(index=False):
        u = int(row.userId); i = int(row.movieId); r = float(row.rating)
        by_user[u][i] = r; by_item[i][u] = r
    user_means = {u:(sum(d.values())/len(d)) if d else 0.0 for u,d in by_user.items()}
    return UIData(by_user, by_item, user_means, set(by_user.keys()), set(by_item.keys()))

def pearson_similarity(u_r: Dict[int,float], v_r: Dict[int,float], mu_u: float, mu_v: float,
                       min_overlap: int = 2, shrink: float = 10.0) -> float:
    common = set(u_r) & set(v_r); n = len(common)
    if n < min_overlap: return 0.0
    num=du2=dv2=0.0
    for i in common:
        du = u_r[i]-mu_u; dv = v_r[i]-mu_v
        num += du*dv; du2 += du*du; dv2 += dv*dv
    if du2<=1e-12 or dv2<=1e-12: return 0.0
    rho = num/math.sqrt(du2*dv2)
    return (n/(n+shrink))*rho  # why: stabilize tiny overlaps

def top_k_neighbors(u: int, ui: UIData, k: int = 25) -> List[Tuple[int,float]]:
    sims=[]
    u_r = ui.by_user.get(u,{})
    mu = ui.user_means.get(u,0.0)
    for v in ui.users:
        if v==u: continue
        s = pearson_similarity(u_r, ui.by_user.get(v,{}), mu, ui.user_means.get(v,0.0))
        if s>0: sims.append((v,s))
    sims.sort(key=lambda x:x[1], reverse=True)
    return sims[:k]

def predict_user_scores(u: int, ui: UIData, k: int = 25) -> Dict[int,float]:
    nbs = top_k_neighbors(u, ui, k)
    seen = set(ui.by_user.get(u,{}))
    numer=defaultdict(float); denom=defaultdict(float)
    for v,s in nbs:
        vm = ui.user_means[v]
        for i,rv in ui.by_user[v].items():
            if i in seen: continue
            numer[i] += s*(rv-vm); denom[i] += abs(s)
    um = ui.user_means.get(u,0.0)
    return {i: um + numer[i]/denom[i] for i in numer if denom[i]>1e-12}

def top_n_for_user(u: int, ui: UIData, movies: pd.DataFrame, n: int = 10, k: int = 25) -> pd.DataFrame:
    preds = predict_user_scores(u, ui, k)
    if not preds: return pd.DataFrame(columns=["movieId","title","score"])
    df = pd.DataFrame([{"movieId":i,"score":s} for i,s in preds.items()])
    return df.merge(movies[["movieId","title","genres","year"]], on="movieId", how="left").sort_values("score", ascending=False).head(n)

# Optional: group recommendation helpers (sequential fairness)
def score_items_for_group(group: List[int], ui: UIData, k: int = 25) -> Dict[int, Dict[int, float]]:
    per_user = {u: predict_user_scores(u, ui, k) for u in group}
    all_items=set().union(*[set(p.keys()) for p in per_user.values()]) if per_user else set()
    out={}
    for i in all_items: out[i] = {u: per_user[u][i] for u in group if i in per_user[u]}
    return out

def rank_group_weighted(scores: Dict[int, Dict[int,float]], movies: pd.DataFrame, group: List[int], weights: List[float]) -> pd.DataFrame:
    rows=[]
    for i,u2s in scores.items():
        num=den=0.0
        for w,u in zip(weights,group):
            if u in u2s: num += w*u2s[u]; den += w
        if den>0:
            mu = num/den
            std = float(np.std(list(u2s.values()))) if len(u2s)>1 else 0.0
            rows.append((i,mu,std))
    df = pd.DataFrame(rows, columns=["movieId","score","disagreement"]).merge(movies[["movieId","title","genres","year"]], on="movieId", how="left")
    return df.sort_values(["score","disagreement"], ascending=[False,True])

def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = (x - x.min())/max(1e-12,(x.max()-x.min())); z = -z/max(1e-6,temperature)
    e = np.exp(z - np.max(z)); return e/np.sum(e)

def sequential_group_recs(group: List[int], ui: UIData, movies: pd.DataFrame, rounds: int = 3, k: int = 25, alpha: float = 1.0, topn: int = 10) -> List[pd.DataFrame]:
    coverage = {u:0.0 for u in group}; used=set(); outs=[]
    for _ in range(rounds):
        scores = score_items_for_group(group, ui, k)
        if used: scores = {i:s for i,s in scores.items() if i not in used}
        cov_vec = np.array([coverage[u] for u in group], float); weights = softmax(alpha*cov_vec)
        ranked = rank_group_weighted(scores, movies, group, weights.tolist())
        topdf = ranked.head(topn).reset_index(drop=True); outs.append(topdf)
        for _,r in topdf.iterrows():
            used.add(int(r["movieId"]))
            for u in group:
                if u in scores.get(int(r["movieId"]), {}):
                    coverage[u] += scores[int(r["movieId"])][u]
    return outs