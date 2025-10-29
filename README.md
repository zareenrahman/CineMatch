# 🎬 CineMatch

**CineMatch** is an interactive movie recommendation app built with **Streamlit** and **Python**, powered by **MovieLens** datasets. It lets you **search any movie**, view its details, and explore **AI-based similar titles** using **cosine similarity and Pearson correlation**.

---

## 🚀 Live Demo

---

## Features

- **Real-time movie search** — case-insensitive, year-aware matching
- **Content-based filtering** — similar titles via cosine similarity on user–item ratings
- **Automatic dataset bootstrap** — downloads MovieLens 100K or 1M if not found locally
- **Movie info panel** — title, genres, year, average rating, number of ratings
- **Interactive UI** — drill down by clicking recommendations
- **Fast caching** — reuses precomputed similarity matrices for efficiency  

---

## Project Structure

```

CineMatch/
├─ dashboard/
│  └─ app.py
├─ recsys/
│  ├─ __init__.py
│  └─ recsys_core.py
├── requirements.txt
├── README.md                 # Project documentation
├── LICENSE                   # Open-source license (e.g. MIT)
└─ streamlit_app.py

````

---

## Core Libraries Used

| Library | Purpose |
|----------|----------|
| **streamlit** | Interactive dashboard for search & recommendations |
| **pandas** | Data manipulation and preprocessing |
| **numpy** | Matrix operations and numerical computation |
| **scipy** | Sparse matrices and cosine similarity computations |
| **requests** | Downloading MovieLens dataset dynamically |
| **dataclasses** | Lightweight data structures for bundle management |
| **zipfile, io, os, re, json** | File I/O, regex parsing, and JSON metadata |

---

## Installation & Local Run

### 1️⃣ Clone or download the repository
If Git is installed:
```bash
git clone https://github.com/zareenrahman/CineMatch.git
cd CineMatch
````

If not, just download the ZIP and extract it.

### Set up virtual environment (recommended)

```bash
python -m venv .venv
.venv\Scripts\activate     # on Windows
# or source .venv/bin/activate  (on macOS/Linux)
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the app

```bash
streamlit run dashboard/app.py
```

Then open in your browser:
👉 [http://localhost:8501](http://localhost:8501)

---

## ☁️ Deploy on Streamlit Cloud

1. Push this repo to GitHub (you already did ✅)
2. Go to [Streamlit Cloud](https://share.streamlit.io)
3. **New App →**

   * Repo: `zareenrahman/CineMatch`
   * Branch: `main`
   * File: `dashboard/app.py`
4. Click **Deploy**

👉 `https://cinematch-zareenrahman.streamlit.app`

---

## Data Source

Movie data is automatically fetched from [**GroupLens MovieLens Datasets**](https://grouplens.org/datasets/movielens/):

* `ml-100k` or `ml-1m` (auto-detected)
* Used for non-commercial academic and demo purposes.
  
---

## Future Enhancements

* [ ] Add collaborative filtering (user-based kNN using Pearson)
* [ ] Include movie posters via TMDB API
* [ ] Dark/light theme toggle
* [ ] Add genre-based filtering and ratings histogram

---

### Quick Start (for reviewers / HR)

> 1. Click the **Live Demo** link
> 2. Try searching for “Toy Story” or “Matrix”
> 3. Click a movie → see similar recommendations
> 4. Explore, enjoy, and see your recommender in action

---
