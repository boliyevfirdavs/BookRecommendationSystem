import numpy as np
import pandas as pd


def build_user_item_matrix(books_path: str, ratings_path: str) -> pd.DataFrame:
    books = pd.read_csv(books_path)[["book_id", "title", "authors"]]
    ratings = pd.read_csv(ratings_path)
    ratings = ratings.merge(books[["book_id", "title"]], how="left", on="book_id").dropna()
    return ratings.pivot_table(index="user_id", columns="title", values="rating").fillna(np.nan)


def user_similarities(uii_matrix: pd.DataFrame, user_id: int, min_common_rated: int = 10) -> pd.Series:
    my_rated = uii_matrix.loc[user_id].notna()
    common_counts = uii_matrix.apply(lambda x: (x.notna() & my_rated).sum(), axis=1)
    sims = uii_matrix[common_counts >= min_common_rated].corrwith(uii_matrix.loc[user_id], axis=1)
    sims[user_id] = np.nan
    return sims


def recommend_books(
    uii_matrix: pd.DataFrame,
    user_id: int,
    top_n: int = 10,
    min_similarity: float = 0.7,
    min_ratings: int = 5,
) -> pd.Series:
    already_read = uii_matrix.loc[user_id].notna()
    sims = user_similarities(uii_matrix, user_id)

    def score(col):
        valid = (sims.reindex(col.index) > min_similarity) & col.notna()
        if valid.sum() <= min_ratings:
            return np.nan
        return col[valid] @ sims.reindex(col.index)[valid] / sims.reindex(col.index)[valid].sum()

    scores = uii_matrix.apply(score)
    return scores[~already_read].sort_values(ascending=False).head(top_n)


if __name__ == "__main__":
    uii_matrix = build_user_item_matrix("books.csv", "ratings.csv")
    print(recommend_books(uii_matrix, user_id=567, top_n=10))