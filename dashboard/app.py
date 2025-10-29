# =========================
# file: dashboard/app.py
# =========================
import os
import streamlit as st
import pandas as pd

from recsys.recsys_core import (
    bootstrap_data, search_movies, movie_card, similar_items,
)

st.set_page_config(page_title="CineMatch", page_icon="🎬", layout="wide")

@st.cache_resource(show_spinner=True)
def _bundle(data_dir: str):
    return bootstrap_data(data_dir)

def _set_selected(mid: int | None, sync_widget: bool = False):
    val = int(mid) if mid is not None else None
    st.session_state["selected_movie_id"] = val
    if sync_widget:
        # consumed before selectbox is rendered on the next run
        st.session_state["pending_pick"] = val

def _get_selected() -> int | None:
    return st.session_state.get("selected_movie_id")

def main():
    st.title("🎬 CineMatch")

    # --- Sidebar / dataset ---
    with st.sidebar:
        st.header("Dataset")
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        st.caption(f"Data dir: `{os.path.abspath(data_dir)}`")
        bundle = _bundle(data_dir)
        st.success(f"Users: {len(bundle.user_map):,} | Movies: {len(bundle.item_map):,}")

        if st.button("↻ Reset cache / recompute"):
            try:
                st.cache_data.clear()
                st.cache_resource.clear()
            except Exception:
                pass
            cache_path = os.path.join(data_dir, "item_sims_topk.npz")
            if os.path.exists(cache_path):
                try: os.remove(cache_path)
                except Exception: pass
            st.toast("Cache cleared. Rerunning…")
            st.rerun()

    # --- Search ---
    st.subheader("Search by movie name (year optional)")
    q = st.text_input(
        "Try: Toy Story  |  Toy Story (1995)  |  Matrix  |  Godfather",
        key="q",
    )

    # Clear selection if the query changed
    last_q = st.session_state.get("last_q")
    if q != last_q:
        st.session_state["last_q"] = q
        st.session_state.pop("pick", None)         # safe: before widget
        st.session_state.pop("pending_pick", None)
        _set_selected(None)

    # Callback for the selectbox so it doesn’t overwrite button clicks
    def _on_pick_change():
        sel = st.session_state.get("pick")
        if sel is not None:
            st.session_state["selected_movie_id"] = int(sel)

    if q.strip():
        res = search_movies(bundle, q, limit=25)
        if res.empty:
            st.warning("No matches. Try another name.")
        else:
            left, right = st.columns([2, 1])
            with left:
                options = res["movieId"].tolist()
                title_lookup = res.set_index("movieId")["title"].to_dict()

                # Determine default value BEFORE rendering the selectbox
                pending = st.session_state.pop("pending_pick", None)
                current = _get_selected()
                pick_state = st.session_state.get("pick")

                if pending in options:
                    default_value = pending
                elif isinstance(pick_state, int) and pick_state in options:
                    default_value = pick_state
                elif current in options:
                    default_value = current
                else:
                    default_value = options[0]
                    _set_selected(default_value)  # keep card consistent

                st.selectbox(
                    "Pick a movie",
                    options=options,
                    index=options.index(default_value),
                    key="pick",
                    format_func=lambda mid: title_lookup.get(mid, str(mid)),
                    on_change=_on_pick_change,
                )
            with right:
                st.dataframe(
                    res.rename(columns={"movieId": "ID"}),
                    hide_index=True,
                    use_container_width=True,
                )

    sel = _get_selected()
    if sel:
        info = movie_card(bundle, sel)
        st.markdown(f"### ✅ {info['title']}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Avg rating", f"{info['avg_rating'] if info['avg_rating'] is not None else '—'}")
        c2.metric("# Ratings", f"{info['n_ratings']:,}")
        c3.metric("Year", f"{info['year'] if info['year'] else '—'}")
        c4.metric("Genres", info['genres'][:35] + ("…" if len(info['genres']) > 35 else ""))

        st.divider()
        st.subheader("Similar titles (item–item)")
        k_sims = st.slider("How many suggestions?", 5, 20, 10, 1, key="k_sims")
        sims = similar_items(bundle, sel, k=k_sims)
        if sims.empty:
            st.info("No similar items found.")
        else:
            bcols = st.columns(5)
            for i, row in sims.head(10).iterrows():
                col = bcols[i % 5]
                label = f"{row['title']}"
                if col.button(label, key=f"sugg_{int(row['movieId'])}"):
                    # schedule selection update for the next run, then rerun
                    _set_selected(int(row["movieId"]), sync_widget=True)
                    st.rerun()

            st.dataframe(
                sims.rename(columns={"movieId": "ID"})[["ID", "title", "genres", "year", "similarity"]],
                hide_index=True,
                use_container_width=True,
            )

if __name__ == "__main__":

    main()
