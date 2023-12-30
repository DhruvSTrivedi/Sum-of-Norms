---

# Convex Clustering Using Sum-of-Norms Approach

## Overview

This document outlines the application of the sum-of-norms clustering method, which is a type of convex optimization problem in unsupervised machine learning. This method is particularly useful for discovering natural groupings within a dataset based on underlying patterns.

## Variables and Functions

### Variables:
- `U`: Cluster centroids for each data point in `X`.
- `F`: Auxiliary variables for the Alternating Direction Method of Multipliers (ADMM) algorithm to ensure the equality constraints.

### Objective Function:
- Data fidelity term: `0.5 * cp.sum_squares(U - X)`
- Regularization term: `gamma * cp.sum(cp.norm(F, axis=1))`

## Constraints and Optimization

### Constraints:
- Enforced by list comprehension: `'[F[i*n + j, :] == U[i, :] - U[j, :] for i in range(n) for j in range(n)]'`
- Ensures differences between centroids are accounted for.

### Optimization Problem:
- Defined as `problem = cp.Problem(cp.Minimize(objective), constraints)`
- Solved using the SCS solver.

## Application to Data

The following Python code snippet applies the sum-of-norms clustering to a smoking dataset:

```python
# Data Standardization and Clustering
clustering_data_scaled = scaler.fit_transform(clustering_data)
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
clusters = kmeans.fit_predict(clustering_data_scaled)

# Data Visualization
plt.scatter(clustering_data_scaled[:, 0], clustering_data_scaled[:, 1], c=clusters, cmap='viridis')
plt.colorbar(label='Cluster')
plt.show()

# Sum-of-Norms Clustering Function
def sum_of_norms_clustering(X, gamma):
    # ... function implementation ...

# Applying the function to the dataset
centroids_subset = sum_of_norms_clustering(X_std_subset, gamma)
```

## Visualization Post-Clustering

The clustered data can be visualized using dimensionality reduction techniques such as PCA:

```python
# PCA for dimensionality reduction
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(clustered_data[...])

# Cluster Visualization
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=clusters, cmap='viridis')
plt.colorbar(label='Cluster ID')
plt.show()
```

Each point represents a data instance, and the color coding corresponds to the cluster assignment.

## Theoretical Background

The sum-of-norms clustering differs from K-means in several key aspects, including the objective function, cluster shapes and sizes, determination of the number of clusters, robustness to outliers, computation complexity, and interoperability.

## Differences
| Factor | K-means Clustering | Sum-of-Norms Clustering |
|--------|--------------------|-------------------------|
| **Objective Function** | Aims to partition data into `K` clusters minimizing within-cluster sum of squares. | Minimizes a combination of the sum of squared differences and a regularization term promoting sparsity in centroid differences. |
| **Cluster Shape and Size** | Assumes clusters are spherical with similar sizes, which can be limiting. | Does not make strong assumptions about shape or size, can find clusters with irregular shapes and hierarchical structures. |
| **Determination of Cluster Number** | Requires the number of clusters (`K`) to be specified a priori. | Can determine the number of clusters based on data and the regularization parameter (`gamma`); a larger `gamma` leads to fewer clusters. |
| **Robustness to Outliers** | Can be sensitive to outliers as they significantly influence the mean of a cluster. | More robust to outliers due to the regularization term promoting similar cluster assignments. |
| **Computation** | Generally faster and scales better to large datasets. | More computationally intensive due to solving a convex optimization problem, especially as dataset size grows. |
| **Interpretability** | Simple to understand and interpret but may oversimplify clustering structure. | Provides a nuanced view of data structure, revealing complex clustering patterns that K-means might miss. |

## Conclusion
Sum-of-norms clustering provides a nuanced view of data structures, revealing complex patterns that simpler algorithms like K-means might not capture. This makes it a powerful tool for unsupervised learning in datasets with intricate groupings. In the above project, the results from sum-of-norms clustering might show a different grouping of data points compared to K-means, potentially with a different number of clusters, cluster shapes, and robustness to outliers. The visual representation provided after sum-of-norms clustering reflects these nuances and can be compared to the K-means results to observe the differences in cluster assignments and structures.


---
