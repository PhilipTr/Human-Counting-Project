import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from hdbscan import HDBSCAN
from collections import Counter
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

# --- Static Object Removal Class ---
class StaticObjectRemover(BaseEstimator, TransformerMixin):
    def __init__(self, frame_col="frame_id", x_col="x", y_col="y", z_col="z", timestamp_col="timestamp",
                 min_std_threshold=5.0, round_digits=2):
        self.frame_col = frame_col
        self.x_col = x_col
        self.y_col = y_col
        self.z_col = z_col
        self.timestamp_col = timestamp_col
        self.min_std_threshold = min_std_threshold  # e.g., 5 seconds
        self.round_digits = round_digits
        self.static_coords = set()

    def fit(self, X, y=None):
        X_rounded = X.copy()
        X_rounded[self.x_col] = X_rounded[self.x_col].round(self.round_digits)
        X_rounded[self.y_col] = X_rounded[self.y_col].round(self.round_digits)
        X_rounded[self.z_col] = X_rounded[self.z_col].round(self.round_digits)
        # Calculate std of timestamps for each coordinate bin
        grouped = X_rounded.groupby([self.x_col, self.y_col, self.z_col])[self.timestamp_col].agg(
            lambda x: x.std() if len(x) > 1 else 0
        )
        
        # Mark as static if std > threshold (spread out over time)
        self.static_coords = set(grouped[grouped > self.min_std_threshold].index)
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy[self.x_col] = X_copy[self.x_col].round(self.round_digits)
        X_copy[self.y_col] = X_copy[self.y_col].round(self.round_digits)
        X_copy[self.z_col] = X_copy[self.z_col].round(self.round_digits)
        
        mask = ~X_copy.apply(lambda row: (row[self.x_col], row[self.y_col], row[self.z_col]) in self.static_coords, axis=1)
        return X[mask]

# --- Noise Filter Class ---
class StatisticalOutlierRemover(BaseEstimator, TransformerMixin):
    """
    Removes noise using Statistical Outlier Removal.
    Points are removed if their average distance to k-nearest neighbors exceeds
    a threshold (mean + alpha * std).
    """
    def __init__(self, k=15, alpha=1.5, debug=False):
        self.k = k  # Number of neighbors
        self.alpha = alpha  # Threshold multiplier
        self.debug = debug

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Use original coordinates for distance calculation
        points = X[["x", "y", "z"]].values
        tree = KDTree(points)
        distances, _ = tree.query(points, k=self.k + 1)  # +1 to exclude the point itself
        avg_distances = np.mean(distances[:, 1:], axis=1)  # Average distance to neighbors
        
        # Calculate threshold
        mean_avg_dist = np.mean(avg_distances)
        std_avg_dist = np.std(avg_distances)
        threshold = mean_avg_dist + self.alpha * std_avg_dist
        
        # Filter points within threshold
        mask = avg_distances <= threshold
        X_filtered = X[mask].copy()
        
        if self.debug:
            print(f"Removed {len(X) - len(X_filtered)} noise points.")
        return X_filtered

# --- Selective Scaler Class ---
class SelectiveScaler(BaseEstimator, TransformerMixin):
    """
    Scales specified columns using StandardScaler while preserving others.
    """
    def __init__(self, columns_to_scale):
        self.columns_to_scale = columns_to_scale
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns_to_scale])
        return self

    def transform(self, X):
        X_scaled = X.copy()
        X_scaled[self.columns_to_scale] = self.scaler.transform(X[self.columns_to_scale])
        return X_scaled

# --- Main Processing ---
# 1. Load radar data (adjust file path as needed)
df = pd.read_csv("Radar_data_filt/new5_filt.txt", header=None, skiprows=1)
df.columns = ["type", "frame_id", "unused_col", "x", "y", "z", "Range", "timestamp"]

# 2. Prepare features
features = df[["frame_id", "x", "y", "z", "Range", "timestamp"]]

# 2. Compute total unique frames
total_frames = df["frame_id"].nunique()
print(f"Total unique frames: {total_frames}")

min_clust_val = int(total_frames*0.25)

# 3. Define the pipeline
pipeline = Pipeline([
    ("static_removal", StaticObjectRemover(
        frame_col="frame_id", x_col="x", y_col="y", z_col="z",
        min_std_threshold=0.5, round_digits=1
    )),
    ("noise_filter", StatisticalOutlierRemover(k=15, alpha=1.5, debug=True)),
    ("scaler", SelectiveScaler(columns_to_scale=["frame_id", "x", "y", "z", "Range"]))
])

# 4. Apply the pipeline
transformed_data = pipeline.fit_transform(features)
X_transformed = pd.DataFrame(transformed_data, columns=features.columns, index=features.index)


# 5. Perform HDBSCAN clustering
X_for_clustering = X_transformed[["frame_id", "x", "y", "z", "Range"]]
hdbscan_clusterer = HDBSCAN(min_cluster_size=min_clust_val, min_samples=5, cluster_selection_epsilon=0.2)
labels = hdbscan_clusterer.fit_predict(X_for_clustering)

# 6. Analyze clustering results
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = np.sum(labels == -1)
print(f"Number of clusters (excluding noise): {n_clusters}")
print(f"Number of noise points: {n_noise}")
for cluster_id, count in Counter(labels).items():
    if cluster_id != -1:
        print(f"Cluster {cluster_id}: {count} points")
print(f"Noise (-1): {Counter(labels)[-1]} points")

# 7. Add labels to data and filter out noise for plotting
X_transformed["cluster_label"] = labels
df_final = X_transformed[X_transformed["cluster_label"] != -1]

# 8. Plot the results
plt.figure(figsize=(10, 8))
plt.scatter(df_final["x"], df_final["y"], c=df_final["cluster_label"], cmap="jet", s=10)
plt.xlabel("x (meters)")
plt.ylabel("y (meters)")
plt.title("Clusters (Static & Noise Removed) Colored by Timestamp")
plt.colorbar(label="Timestamp")
plt.show()