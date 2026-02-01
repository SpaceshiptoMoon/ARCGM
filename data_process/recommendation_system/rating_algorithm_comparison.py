import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.decomposition import randomized_svd
from tabulate import tabulate


def load_rating_data():
    rate_data = pd.read_csv("rating_fillna.csv")
    rate_mask = pd.read_csv("rating_mask.csv")
    return rate_data, rate_mask


def prepare_train_data():
    rate_data, _ = load_rating_data()

    train_data = []
    for index, row in rate_data.iterrows():
        user_id = row['userId']
        for item_id in rate_data.columns[1:]:
            rating = row[item_id]
            train_data.append([user_id, int(item_id), rating])

    train_data = np.array(train_data).astype(int)

    print("Train data shape:", train_data.shape)
    print("First few rows:")
    for i in range(5):
        print(train_data[i])

    return train_data


class MF:
    def __init__(self, n_factors=20, learning_rate=0.01, reg=0.02, iterations=100):
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.reg = reg
        self.iterations = iterations

    def fit(self, train_data):
        users, items, ratings = train_data[:, 0], train_data[:, 1], train_data[:, 2]
        n_users = int(np.max(users)) + 1
        n_items = int(np.max(items)) + 1

        self.P = np.random.normal(scale=1./self.n_factors, size=(n_users, self.n_factors))
        self.Q = np.random.normal(scale=1./self.n_factors, size=(n_items, self.n_factors))

        for _ in range(self.iterations):
            for u, i, r_ui in zip(users, items, ratings):
                error = r_ui - np.dot(self.P[u], self.Q[i])
                self.P[u] += self.learning_rate * (error * self.Q[i] - self.reg * self.P[u])
                self.Q[i] += self.learning_rate * (error * self.P[u] - self.reg * self.Q[i])

    def predict(self, user, item):
        return np.dot(self.P[user], self.Q[item])


class BiasSVD:
    def __init__(self, n_factors=20, learning_rate=0.01, reg=0.02, iterations=100):
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.reg = reg
        self.iterations = iterations

    def fit(self, train_data):
        users, items, ratings = train_data[:, 0], train_data[:, 1], train_data[:, 2]
        n_users = int(np.max(users)) + 1
        n_items = int(np.max(items)) + 1

        self.global_mean = np.mean(ratings)
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)
        self.P = np.random.normal(scale=1./self.n_factors, size=(n_users, self.n_factors))
        self.Q = np.random.normal(scale=1./self.n_factors, size=(n_items, self.n_factors))

        for _ in range(self.iterations):
            for u, i, r_ui in zip(users, items, ratings):
                error = r_ui - (self.global_mean + self.user_bias[u] + self.item_bias[i] + np.dot(self.P[u], self.Q[i]))
                self.user_bias[u] += self.learning_rate * (error - self.reg * self.user_bias[u])
                self.item_bias[i] += self.learning_rate * (error - self.reg * self.item_bias[i])
                self.P[u] += self.learning_rate * (error * self.Q[i] - self.reg * self.P[u])
                self.Q[i] += self.learning_rate * (error * self.P[u] - self.reg * self.Q[i])

    def predict(self, user, item):
        return self.global_mean + self.user_bias[user] + self.item_bias[item] + np.dot(self.P[user], self.Q[item])


class SVDpp:
    def __init__(self, n_factors=20, learning_rate=0.01, reg=0.02, iterations=50):
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.reg = reg
        self.iterations = iterations

    def fit(self, train_data):
        users, items, ratings = train_data[:, 0], train_data[:, 1], train_data[:, 2]
        n_users = int(np.max(users)) + 1
        n_items = int(np.max(items)) + 1

        self.global_mean = np.mean(ratings)
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)
        self.P = np.random.normal(scale=1./self.n_factors, size=(n_users, self.n_factors))
        self.Q = np.random.normal(scale=1./self.n_factors, size=(n_items, self.n_factors))
        self.Y = np.random.normal(scale=1./self.n_factors, size=(n_items, self.n_factors))

        self.user_history = {}
        for u, i, _ in train_data:
            if u not in self.user_history:
                self.user_history[u] = []
            self.user_history[u].append(i)

        for _ in range(self.iterations):
            for u, i, r_ui in zip(users, items, ratings):
                y_sum = np.zeros(self.n_factors)
                if u in self.user_history:
                    for j in self.user_history[u]:
                        y_sum += self.Y[j]
                    y_avg = y_sum / len(self.user_history[u])
                else:
                    y_avg = np.zeros(self.n_factors)

                pred = self.global_mean + self.user_bias[u] + self.item_bias[i] + np.dot(self.P[u], self.Q[i] + y_avg)
                error = r_ui - pred

                self.user_bias[u] += self.learning_rate * (error - self.reg * self.user_bias[u])
                self.item_bias[i] += self.learning_rate * (error - self.reg * self.item_bias[i])
                self.P[u] += self.learning_rate * (error * (self.Q[i] + y_avg) - self.reg * self.P[u])
                self.Q[i] += self.learning_rate * (error * self.P[u] - self.reg * self.Q[i])

                if u in self.user_history:
                    for j in self.user_history[u]:
                        self.Y[j] += self.learning_rate * (error * self.P[u] / len(self.user_history[u]) - self.reg * self.Y[j])

    def predict(self, user, item):
        y_sum = np.zeros(self.n_factors)
        if user in self.user_history:
            for j in self.user_history[user]:
                y_sum += self.Y[j]
            y_avg = y_sum / len(self.user_history[user])
        else:
            y_avg = np.zeros(self.n_factors)

        return self.global_mean + self.user_bias[user] + self.item_bias[item] + np.dot(self.P[user], self.Q[item] + y_avg)


def svd_impute_zeros(X, rank=50, max_iter=50, verbose=False):
    missing_mask = (X == 0)

    col_means = np.where(np.any(missing_mask, axis=0),
                          np.sum(X, axis=0) / np.count_nonzero(X, axis=0),
                          np.mean(X, axis=0))
    X_filled = X.copy()
    for i in range(X.shape[1]):
        if np.any(missing_mask[:, i]):
            X_filled[missing_mask[:, i], i] = col_means[i]

    for iter_num in range(max_iter):
        U, Sigma, Vt = randomized_svd(X_filled, n_components=rank)

        X_reconstructed = U @ np.diag(Sigma) @ Vt

        X_filled[missing_mask] = X_reconstructed[missing_mask]

        if verbose:
            print(f"Iteration {iter_num + 1}, MSE on zeros: "
                  f"{np.nanmean((X_filled[missing_mask] - X_reconstructed[missing_mask]) ** 2):.6f}")

    return X_filled


def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def calculate_mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)


if __name__ == "__main__":
    print("=" * 60)
    print("推荐系统评分算法对比 - Recommendation System Rating Algorithm Comparison")
    print("=" * 60)

    train_data = prepare_train_data()

    print("\n" + "=" * 40)
    print("Basic Model Predictions:")
    print("=" * 40)

    print("\n1. Matrix Factorization (MF):")
    mf = MF(n_factors=20, iterations=10)
    mf.fit(train_data)
    mf_pred = mf.predict(0, 2)
    print(f"MF 预测评分 (MF Prediction): {mf_pred:.4f}")

    print("\n2. Bias-SVD:")
    bias_svd = BiasSVD(n_factors=20, iterations=10)
    bias_svd.fit(train_data)
    bias_svd_pred = bias_svd.predict(0, 2)
    print(f"Bias-SVD 预测评分 (Bias-SVD Prediction): {bias_svd_pred:.4f}")

    print("\n3. SVD++:")
    svdpp = SVDpp(n_factors=20, iterations=10)
    try:
        svdpp.fit(train_data)
        svdpp_pred = svdpp.predict(0, 2)
        print(f"SVD++ 预测评分 (SVD++ Prediction): {svdpp_pred:.4f}")
    except KeyboardInterrupt:
        print("SVD++ training interrupted")

    print("\n" + "=" * 40)
    print("程序结束 (Program End)")
    print("=" * 40)