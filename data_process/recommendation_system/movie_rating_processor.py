import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def load_ratings_data(ratings_path='ratings.csv'):
    ratings_df = pd.read_csv(ratings_path)
    return ratings_df


def create_rating_matrix(ratings_df, top_n_movies=1000):
    rating_matrix = ratings_df.pivot_table(
        index='userId',
        columns='movieId',
        values='rating'
    )

    print(f"Original rating matrix shape: {rating_matrix.shape}")

    rating_matrix = rating_matrix.fillna(0)

    if top_n_movies > 0:
        movie_ratings_count = (rating_matrix != 0).sum(axis=0)
        top_movies = movie_ratings_count.nlargest(top_n_movies).index
        rating_matrix = rating_matrix[top_movies]

    print(f"Final rating matrix shape: {rating_matrix.shape}")

    return rating_matrix


def visualize_missing_values(rating_matrix, figsize=(12, 8)):
    plt.figure(figsize=figsize)
    sns.heatmap(rating_matrix.isnull(), cbar=False, cmap='viridis')
    plt.title("User-Movie Rating Matrix - Missing Values", fontsize=14, pad=20)
    plt.xlabel("MovieID", fontsize=12)
    plt.ylabel("UserID", fontsize=12)
    plt.tight_layout()
    plt.show()

    print(f"\nMissing values in original matrix: {rating_matrix.isnull().sum().sum()}")


def create_rating_mask(rating_matrix):
    rating_mask = (rating_matrix != 0).astype(int)

    print(f"Rating mask shape: {rating_mask.shape}")
    print(f"Number of ratings (1s): {rating_mask.sum().sum()}")

    return rating_mask


def save_matrices(rating_matrix, rating_mask, rating_fillna_path="rating_fillna.csv",
                 rating_mask_path="rating_mask.csv"):
    rating_matrix.to_csv(rating_fillna_path)
    print(f"Rating matrix saved to: {rating_fillna_path}")

    rating_mask.to_csv(rating_mask_path)
    print(f"Rating mask saved to: {rating_mask_path}")


def analyze_rating_matrix(rating_matrix):
    print("\n" + "="*50)
    print("Rating Matrix Analysis")
    print("="*50)

    print(f"Matrix dimensions: {rating_matrix.shape[0]} users × {rating_matrix.shape[1]} movies")
    print(f"Total cells: {rating_matrix.shape[0] * rating_matrix.shape[1]}")
    print(f"Non-zero ratings: {(rating_matrix != 0).sum().sum()}")
    print(f"Sparse ratio: {(rating_matrix == 0).sum().sum() / (rating_matrix.shape[0] * rating_matrix.shape[1]) * 100:.2f}%")

    ratings = rating_matrix[rating_matrix != 0]
    print(f"\nRating distribution:")
    print(ratings.value_counts().sort_index())

    user_ratings = (rating_matrix != 0).sum(axis=1)
    print(f"\nUser statistics:")
    print(f"Min ratings per user: {user_ratings.min()}")
    print(f"Max ratings per user: {user_ratings.max()}")
    print(f"Avg ratings per user: {user_ratings.mean():.2f}")

    movie_ratings = (rating_matrix != 0).sum(axis=0)
    print(f"\nMovie statistics:")
    print(f"Min ratings per movie: {movie_ratings.min()}")
    print(f"Max ratings per movie: {movie_ratings.max()}")
    print(f"Avg ratings per movie: {movie_ratings.mean():.2f}")


def main():
    print("=" * 60)
    print("Movie Rating System Data Processing")
    print("电影评分系统数据处理")
    print("=" * 60)

    print("\n1. Loading ratings data...")
    ratings_df = load_ratings_data()
    print(f"Loaded {len(ratings_df)} ratings")
    print(f"Users: {ratings_df['userId'].nunique()}")
    print(f"Movies: {ratings_df['movieId'].nunique()}")

    print("\n2. Creating rating matrix...")
    rating_matrix = create_rating_matrix(ratings_df, top_n_movies=1000)

    print("\n3. Visualizing missing values...")
    visualize_missing_values(rating_matrix)

    analyze_rating_matrix(rating_matrix)

    print("\n4. Creating rating mask...")
    rating_mask = create_rating_mask(rating_matrix)

    print("\n5. Saving matrices...")
    save_matrices(rating_matrix, rating_mask)

    print("\n6. Final matrices:")
    print("\nRating Matrix (first 5 users, first 10 movies):")
    print(rating_matrix.iloc[:5, :10])

    print("\nRating Mask (first 5 users, first 10 movies):")
    print(rating_mask.iloc[:5, :10])

    print("\n" + "="*60)
    print("Processing completed successfully!")
    print("数据处理完成！")
    print("="*60)


if __name__ == "__main__":
    main()