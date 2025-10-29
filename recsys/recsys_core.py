# ==========================================
# file: recsys/recsys_core.py
# ==========================================
from __future__ import annotations

import io
import json
import os
import re
import time
import zipfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry
from scipy import sparse


# ---------- Download sources ----------
ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
ML100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _http_session() -> requests.Session:
    """Retries mitigate transient SSL/network hiccups on Streamlit Cloud."""
    s = requests.Session()
    retries = Retry(
        total=5, backoff_factor=0.8, status_forcelist=[429, 500, 502, 503, 504]
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.headers.update({"User-Agent": "CineMatch/1.0 (+streamlit-cloud)"})
    return s


def _download_zip(url: str, dest_dir: str) -> None:
    _ensure_dir(dest_dir)
    sess = _http_session()
    with sess.get(url, timeout=120, stream=True) as r:
        r.raise_for_status()
        bio = io.BytesIO()
        for chunk in r.iter_content(chunk_size=1024 * 256):
            if chunk:
                bio.write(chunk)
        bio.seek(0)
        with zipfile.ZipFile(bio) as zf:
            zf.extractall(dest_dir)  # why: avoids partial files


def _has_1m(data_dir: str) -> bool:
    base = os.path.join(data_dir, "ml-1m")
    return os.path.exists(os.path.join(base, "ratings.dat")) and os.path.exists(
        os.path.join(base, "movies.dat")
    )


def _has_100k(data_dir: str) -> bool:
    base = os.path.join(data_dir, "ml-100k")
    return os.path.exists(os.path.join(base, "u.data")) and os.path.exists(
        os.path.join(base, "u.item")
    )


def _bootstrap(data_dir: str, force_flavor: str = "auto") -> str:
    """
    Only 1M and 100K. force_flavor in {"auto","ml-1m","ml-100k"}.
    """
    if force_flavor not in {"auto", "ml-1m", "ml-100k"}:
        force_flavor = "auto"

    # Honor local copies first
    if force_flavor in {"auto", "ml-1m"} and _has_1m(data_dir):
        return "ml-1m"
    if force_flavor in {"auto", "ml-100k"} and _has_100k(data_dir):
        return "ml-100k"

    # Download per force
    if force_flavor == "ml-1m":
        try:
            _download_zip(ML1M_URL, data_dir)
            return "ml-1m"
        except Exception:
            pass
        _download_zip(ML100K_URL, data_dir)
        return "ml-100k"

    if force_flavor == "ml-100k":
        _download_zip(ML100K_URL, data_dir)
        return "ml-100k"

    # auto: try 1M then 100K
    try:
        _download_zip(ML1M_URL, data_dir)
        return "ml-1m"
    except Exception:
        pass
    _download_zip(ML100K_URL, data_dir)
    return "ml-100k"


# ---------- loaders ----------
def _load_ml1m(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    base = os.path.join(data_dir, "ml-1m")
    ratings = pd.read_csv(
        os.path.join(base, "ratings.dat"),
        sep="::",
        engine="python",
        header=None,
        names=["userId", "movieId", "rating", "timestamp"],
    )
    movies = pd.read_csv(
        os.path.join(base, "movies.dat"),
        sep="::",
        engine="python",
        header=None,
        encoding="latin-1",
        names=["movieId", "title", "genres"],
    )
    return ratings, movies


def _load_ml100k(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    base = os.path.join(data_dir, "ml-100k")
    ratings = pd.read_csv(
        os.path.join(base, "u.data"),
        sep="\t",
        engine="python",
        header=None,
        names=["userId", "movieId", "rating", "timestamp"],
    )
    raw = pd.read_csv(
        os.path.join(base, "u.item"),
        sep="|",
        engine="python",
        header=None,
        encoding="latin-1",
    )
    raw = raw.rename(columns={0: "movieId", 1: "title"})
    if raw.shape[1] > 5:
        genre_cols = list(range(5, raw.shape[1]))
        genre_names = [
            "unknown",
            "Action",
            "Adventure",
            "Animation",
            "Children's",
            "Comedy",
            "Crime",
            "Documentary",
            "Drama",
            "Fantasy",
            "Film-Noir",
            "Horror",
            "Musical",
            "Mystery",
            "Romance",
            "Sci-Fi",
            "Thriller",
            "War",
            "Western",
        ][: len(genre_cols)]
        gdf = raw[genre_cols]
        genres = gdf.apply(
            lambda r: "|".join([g for g, v in zip(genre_names, r) if int(v) == 1]),
            axis=1,
        )
    else:
        genres = pd.Series([""] * len(raw), index=raw.index)
    movies = pd.DataFrame(
        {"movieId": raw["movieId"], "title": raw["title"], "genres": genres}
    )
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
        try:
            year = int(m.group(1))
        except Exception:
            year = None
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
    return (
        g.mean()
        .to_frame("avg_rating")
        .join(g.count().to_frame("n_ratings"))
        .reset_index()
    )


# ---------- item-item similarity ----------
def _build_maps(ratings: pd.DataFrame, movies: pd.DataFrame):
    uids = np.sort(ratings["userId"].unique())
    iids = np.sort(movies["movieId"].unique())
    u_map = {u: i for i, u in enumerate(uids)}
    i_map = {m: j for j, m in enumerate(iids)}
    return u_map, i_map, {i: u for u, i in u_map.items()}, {j: m for m, j in i_map.items()}


def _matrix(ratings: pd.DataFrame, u_map, i_map) -> sparse.csr_matrix:
    rows = ratings["userId"].map(u_map).values
    cols = ratings["movieId"].map(i_map).values
    vals = ratings["rating"].astype(np.float32).values
    return (
        sparse.coo_matrix((vals, (rows, cols)), shape=(len(u_map), len(i_map)))
        .tocsr()
        .astype(np.float32)
    )


def _binarize(R: sparse.csr_matrix, thr: float = 4.0) -> sparse.csr_matrix:
    data = (R.data >= thr).astype(np.float32)
    return sparse.csr_matrix((data, R.indices, R.indptr), shape=R.shape)


def _topk_item_cosine(R_bin: sparse.csr_matrix, k: int = 50) -> sparse.csr_matrix:
    n_items = R_bin.shape[1]
    norms = np.sqrt(R_bin.multiply(R_bin).sum(axis=0)).A1 + 1e-8
    indptr = [0]
    indices: List[int] = []
    data: List[float] = []
    block = 1024
    Rt = R_bin.T.tocsr()
    for start in range(0, n_items, block):
        stop = min(start + block, n_items)
        numer = Rt @ R_bin[:, start:stop]
        denom = norms[:, None] * norms[start:stop][None, :]
        cs = numer.multiply(sparse.csr_matrix(1.0 / denom))
        # zero self-sim on the overlapping diagonal part
        diag_len = min(cs.shape[0], cs.shape[1])
        if diag_len > 0:
            cs.setdiag(0)
        for j in range(cs.shape[1]):
            col = cs.getcol(j).tocoo()
            if col.nnz == 0:
                indptr.append(indptr[-1])
                continue
            if col.nnz > k:
                top = np.argpartition(-col.data, k - 1)[:k]
                idx = col.row[top]
                val = col.data[top]
                order = np.argsort(-val)
                idx = idx[order]
                val = val[order]
            else:
                idx = col.row
                val = col.data
                order = np.argsort(-val)
                idx = idx[order]
                val = val[order]
            indices.extend(idx.tolist())
            data.extend(val.tolist())
            indptr.append(indptr[-1] + len(idx))
    return sparse.csr_matrix(
        (
            np.array(data, np.float32),
            np.array(indices, np.int32),
            np.array(indptr, np.int32),
        ),
        shape=(n_items, n_items),
    )


def _save_sparse(path: str, mat: sparse.csr_matrix, meta: dict) -> None:
    _ensure_dir(os.path.dirname(path))
    np.savez_compressed(
        path,
        data=mat.data,
        indices=mat.indices,
        indptr=mat.indptr,
        shape=mat.shape,
        meta=json.dumps(meta),
    )


def _load_sparse(path: str) -> Tuple[sparse.csr_matrix, dict]:
    z = np.load(path, allow_pickle=False)
    mat = sparse.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    raw_meta = z["meta"]
    # why: numpy may return np.bytes_ on some stacks
    if isinstance(raw_meta, (bytes, bytearray)):
        meta = json.loads(raw_meta.decode("utf-8"))
    else:
        meta = json.loads(str(raw_meta))
    return mat, meta


# ---------- CF data bundle ----------
@dataclass
class DataBundle:
    data_dir: str
    flavor: str
    ratings: pd.DataFrame
    movies: pd.DataFrame
    user_map: Dict[int, int]
    item_map: Dict[int, int]
    inv_user_map: Dict[int, int]
    inv_item_map: Dict[int, int]
    R: sparse.csr_matrix
    item_sims: sparse.csr_matrix
    item_stats: pd.DataFrame


def _filter_popular_items(
    ratings: pd.DataFrame, min_ratings: int, max_items: int
) -> set[int]:
    counts = ratings.groupby("movieId")["rating"].count().sort_values(ascending=False)
    keep = counts[counts >= min_ratings].head(max_items).index
    return set(map(int, keep))


def bootstrap_data(
    data_dir: str = "data",
    min_ratings_for_sims: int = 50,
    max_items_for_sims: int = 20000,
    topk: int = 50,
    force_flavor: str = "auto",  # "auto" | "ml-1m" | "ml-100k"
    limit_1m: bool = True,  # cap popular items on 1M for speed
) -> DataBundle:
    _ensure_dir(data_dir)
    flavor = _bootstrap(data_dir, force_flavor=force_flavor)

    if flavor == "ml-1m":
        ratings, movies = _load_ml1m(data_dir)
    else:
        ratings, movies = _load_ml100k(data_dir)

    movies = add_title_features(movies)

    # Optional filtering for similarity matrix on 1M
    if flavor == "ml-1m" and limit_1m:
        active_item_ids = _filter_popular_items(
            ratings, min_ratings_for_sims, max_items_for_sims
        )
        ratings = ratings[ratings["movieId"].isin(active_item_ids)].copy()
        movies = movies[movies["movieId"].isin(active_item_ids)].copy()

    user_map, item_map, inv_user_map, inv_item_map = _build_maps(ratings, movies)
    ratings = ratings[ratings["movieId"].isin(item_map)].copy()
    R = _matrix(ratings, user_map, item_map)

    cache = os.path.join(data_dir, f"item_sims_topk_{topk}.npz")
    sims = None
    if os.path.exists(cache):
        try:
            sims, meta = _load_sparse(cache)
            ok = (
                meta.get("shape") == list(R.shape)
                and meta.get("flavor") == flavor
                and meta.get("topk") == topk
                and meta.get("min_ratings_for_sims") == min_ratings_for_sims
                and meta.get("max_items_for_sims") == max_items_for_sims
                and meta.get("limit_1m") == limit_1m
            )
            if not ok:
                sims = None
        except Exception:
            sims = None

    if sims is None:
        Rb = _binarize(R, 4.0)
        sims = _topk_item_cosine(Rb, k=topk)
        _save_sparse(
            cache,
            sims,
            {
                "shape": list(R.shape),
                "flavor": flavor,
                "topk": topk,
                "min_ratings_for_sims": min_ratings_for_sims,
                "max_items_for_sims": max_items_for_sims,
                "limit_1m": limit_1m,
            },
        )

    return DataBundle(
        data_dir=data_dir,
        flavor=flavor,
        ratings=ratings,
        movies=movies,
        user_map=user_map,
        item_map=item_map,
        inv_user_map=inv_user_map,
        inv_item_map=inv_item_map,
        R=R,
        item_sims=sims,
        item_stats=_stats(ratings),
    )


# ---------- search + item info ----------
_YEAR_NUMBER = re.compile(r"\b(\d{4})\b")


def search_movies(bundle: DataBundle, query: str, limit: int = 25) -> pd.DataFrame:
    q_raw = (query or "").strip()
    if not q_raw:
        return pd.DataFrame(columns=["movieId", "title", "genres", "year"])
    m = _YEAR_NUMBER.search(q_raw)
    q_year = int(m.group(1)) if m else None
    q = _YEAR_NUMBER.sub("", q_raw).strip()
    df = bundle.movies.copy()
    mask = df["title"].str.contains(q, case=False, na=False, regex=False) | df[
        "clean_title"
    ].str.contains(q, case=False, na=False, regex=False)
    out = df[mask].copy()
    if q_year:
        out["_year_bias"] = (out["year"] == q_year).astype(int)
        out = out.sort_values(["_year_bias", "title"], ascending=[False, True]).drop(
            columns=["_year_bias"]
        )
    else:
        out = out.sort_values("title")
    return out.head(limit)[["movieId", "title", "genres", "year"]]


def similar_items(bundle: DataBundle, movie_id: int, k: int = 10) -> pd.DataFrame:
    if movie_id not in bundle.item_map:
        return pd.DataFrame(columns=["movieId", "title", "genres", "year", "similarity"])
    j = bundle.item_map[movie_id]
    row = bundle.item_sims.getrow(j).tocoo()
    if row.nnz == 0:
        return pd.DataFrame(columns=["movieId", "title", "genres", "year", "similarity"])
    order = np.argsort(-row.data)
    idx = row.col[order]
    val = row.data[order]
    mids = [bundle.inv_item_map[i] for i in idx if i != j][:k]
    sims = [v for i, v in zip(idx, val) if i != j][:k]
    if not mids:
        return pd.DataFrame(columns=["movieId", "title", "genres", "year", "similarity"])
    info = (
        bundle.movies.set_index("movieId")
        .loc[mids][["title", "genres", "year"]]
        .reset_index()
    )
    sim = (np.array(sims) - float(np.min(sims))) / (
        float(np.max(sims)) - float(np.min(sims)) + 1e-8
    )
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
        "genres": str(row.get("genres", "") or ""),
        "avg_rating": None if np.isnan(avg) else round(avg, 2),
        "n_ratings": cnt,
        "available": True,
    }
