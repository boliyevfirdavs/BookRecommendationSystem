# Book Recommendation System

A collaborative filtering book recommendation system using user-based similarity on the Goodreads dataset.

## How It Works

1. **Build user-item matrix** — Merges `books.csv` and `ratings.csv` into a pivot table of users vs. book titles with ratings as values.
2. **Compute user similarities** — Uses Pearson correlation between a target user and all other users who share at least `min_common_rated` books in common.
3. **Generate recommendations** — Scores unread books using a weighted average of similar users' ratings (filtered by a minimum similarity threshold), then returns the top N.

## Dataset

| File | Description |
|------|-------------|
| `books.csv` | 10,000 books with metadata (title, author, ISBN, ratings, cover URL) |
| `ratings.csv` | User-book ratings (user_id, book_id, rating 1–5) |

## Usage

```bash
python main.py
```

This runs recommendations for `user_id=567` and prints the top 10 suggested books.

To use it in your own code:

```python
from main import build_user_item_matrix, recommend_books

uii_matrix = build_user_item_matrix("books.csv", "ratings.csv")
recommendations = recommend_books(uii_matrix, user_id=567, top_n=10)
print(recommendations)
```

## API

### `build_user_item_matrix(books_path, ratings_path) -> pd.DataFrame`
Returns a pivot table with users as rows and book titles as columns.

### `user_similarities(uii_matrix, user_id, min_common_rated=10) -> pd.Series`
Returns Pearson correlation scores between the target user and all others with enough books in common.

### `recommend_books(uii_matrix, user_id, top_n=10, min_similarity=0.7, min_ratings=5) -> pd.Series`
Returns up to `top_n` unread books scored by similarity-weighted average rating.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `top_n` | 10 | Number of books to recommend |
| `min_similarity` | 0.7 | Minimum Pearson correlation to include a user |
| `min_ratings` | 5 | Minimum number of similar users required to score a book |

## Requirements

- Python 3.x
- pandas
- numpy

Install dependencies:

```bash
pip install pandas numpy
```