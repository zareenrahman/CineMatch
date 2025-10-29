import os
import math
import streamlit as st
import pandas as pd

from recsys.recsys_core import (
    bootstrap_data, search_movies, movie_card, similar_items,
)

st.set_page_config(page_title="CineMatch", page_icon="🎬", layout="wide")

# ---------- Minimal CSS (simple chips + clean buttons) ----------
st.markdown("""
<style>
.block-container {padding-top:1.5rem; padding-bottom:2.5rem;}
.cm-chip {
  display:inline-block; padding:.18rem .5rem; margin:.15rem .25rem .15rem 0;
  border-radius:999px; font-size:.8rem; background: rgba(127,127,127,.08);
  border:1px solid rgba(0,0,0,.12);
}
.cm-title-btn .stButton>button{
  width:100%; text-align:left; font-weight:600; border-radius:12px;
  border:1px solid rgba(0,0,0,.12); background:rgba(255,255,255,.6);
}
.cm-title-btn .stButton>button:hover{box-shadow:0 6px 16px rgba(0,0,0,.08);}
[data-testid="stDataFrame"]{border-radius:12px; overflow:hidden;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=True)
def _bundle(data_dir: str):
    return bootstrap_data(
        data_dir=data_dir,
        min_ratings_for_sims=50,
        max_items_for_sims=20000,
        topk=50,
    )

def _set_selected(mid: int | None, sync_widget: bool = False):
    val = int(mid) if mid is not None else None
    st.session_state["selected_movie_id"] = val
    if sync_widget:
        st.session_state["pending_pick"] = val  # sync selectbox next run

def _get_selected() -> int | None:
    return st.session_state.get("selected_movie_id")

def genre_chips_simple(genres: str) -> str:
    parts = [g for g in (genres or "").split("|") if g.strip()]
    return "".join([f"<span class='cm-chip'>{p}</span>" for p in parts[:10]]) or "<span class='cm-chip'>—</span>"

def main():
    st.markdown("# 🎬 CineMatch")
    st.caption("Search a movie (year optional)")

    # Sidebar
    with st.sidebar:
        st.header("Dataset")
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        st.caption(f"Data dir: `{os.path.abspath(data_dir)}`")
        bundle = _bundle(data_dir)
        st.success(f"Flavor: **{bundle.flavor}**  •  Users: {len(bundle.user_map):,}  •  Movies: {len(bundle.item_map):,}")

        if st.button("↻ Reset cache / recompute"):
            try:
                st.cache_data.clear(); st.cache_resource.clear()
            except Exception:
                pass
            cache_path = os.path.join(data_dir, "item_sims_topk.npz")
            if os.path.exists(cache_path):
                try: os.remove(cache_path)
                except Exception: pass
            st.toast("Cache cleared. Re-running …")
            st.rerun()

    # Search
    st.subheader("Search by movie name (year optional)")
    q = st.text_input("Try: **Toy Story** | **Toy Story (1995)** | **Matrix** | **Godfather**", key="q")

    last_q = st.session_state.get("last_q")
    if q != last_q:
        st.session_state["last_q"] = q
        st.session_state.pop("pick", None)
        st.session_state.pop("pending_pick", None)
        _set_selected(None)

    def _on_pick_change():
        sel = st.session_state.get("pick")
        if sel is not None:
            _set_selected(sel)

    if q.strip():
        res = search_movies(bundle, q, limit=25)
        if res.empty:
            st.warning("No matches. Try another name.")
        else:
            left, right = st.columns([2.5, 1])
            with left:
                options = res["movieId"].tolist()
                t_lookup = res.set_index("movieId")["title"].to_dict()
                pending = st.session_state.pop("pending_pick", None)
                current = _get_selected()
                pick_state = st.session_state.get("pick")

                if pending in options: default_value = pending
                elif isinstance(pick_state, int) and pick_state in options: default_value = pick_state
                elif current in options: default_value = current
                else: default_value = options[0]

                if current is None and pending is None and pick_state is None:
                    _set_selected(default_value, sync_widget=True)

                st.selectbox(
                    "Pick a movie",
                    options=options,
                    index=options.index(default_value),
                    key="pick",
                    format_func=lambda mid: t_lookup.get(mid, str(mid)),
                    on_change=_on_pick_change,
                )
            with right:
                st.dataframe(
                    res.rename(columns={"movieId": "ID"}),
                    hide_index=True,
                    use_container_width=True,
                    height=180,
                )

    # Selected card + similar
    sel = _get_selected()
    if sel:
        info = movie_card(bundle, sel)
        c1, c2 = st.columns([1.1, 2.2], gap="large")

        with c1:
            st.markdown(f"### ✅ {info['title']}")
            m1, m2, m3 = st.columns(3)
            m1.metric("Avg rating", f"{info['avg_rating'] if info['avg_rating'] is not None else '—'}")
            m2.metric("Ratings", f"{info['n_ratings']:,}")
            m3.metric("Year", f"{info['year'] if info['year'] else '—'}")
            st.markdown(genre_chips_simple(info["genres"]), unsafe_allow_html=True)

        with c2:
            st.subheader("Similar titles (item–item)")
            k_sims = st.slider("How many suggestions?", 5, 20, 12, 1, key="k_sims")
            sims = similar_items(bundle, sel, k=k_sims)  # uses getcol().tocoo() + drop self
            if sims.empty:
                st.info("No similar items found.")
            else:
                # Title-only buttons in a grid
                n = min(k_sims, len(sims))
                cols_per_row = 5
                rows = math.ceil(n / cols_per_row)
                for r in range(rows):
                    cols = st.columns(cols_per_row)
                    for c in range(cols_per_row):
                        idx = r * cols_per_row + c
                        if idx >= n: break
                        row = sims.iloc[idx]
                        mid = int(row["movieId"])
                        title = str(row["title"])
                        with cols[c]:
                            with st.container():
                                st.markdown("<div class='cm-title-btn'>", unsafe_allow_html=True)
                                if st.button(title, key=f"sugg_{mid}", use_container_width=True):
                                    _set_selected(mid, sync_widget=False)
                                    st.rerun()
                                st.markdown("</div>", unsafe_allow_html=True)

                st.dataframe(
                    sims.head(n).rename(columns={"movieId": "ID"})[["ID", "title", "genres", "year", "similarity"]],
                    hide_index=True,
                    use_container_width=True,
                    height=200 + 22 * n,
                )

if __name__ == "__main__":
    main()