---

# Convex Clustering Using Sum-of-Norms Approach

## Overview

Convex optimization is an essential subfield of optimization in machine learning, particularly for unsupervised learning tasks such as clustering. It involves the process where the objective function is convex, which simplifies the optimization process by ensuring that every local minimum is also a global minimum. This characteristic makes convex optimization problems generally more straightforward to solve than non-convex optimization problems.

In the context of unsupervised machine learning, convex optimization is leveraged to find the best grouping of data points by minimizing a cost function. Clustering algorithms, including k-means and sum-of-norms clustering, utilize convex optimization to efficiently discover clusters within a dataset. Unlike supervised learning, where models are trained with labelled data, unsupervised learning algorithms like clustering infer the natural structure present within a dataset without prior knowledge of labels.

For more detailed output and code, see the Jupyter Notebook:  
[Notebook](Kmeans vs. Sum of Norms.ipynb)

The research and methods discussed here are based on the work of **[Dr. Stephen Vavasis]**(https://uwaterloo.ca/combinatorics-and-optimization/contacts/stephen-vavasis). For further reading, please refer to his research paper:  
[Certifying clusters from sum-of-norms clustering](https://arxiv.org/abs/2006.11355)

## Convex Clustering Variables and Objective Function

### Variables
- `U`: Represents the cluster centroids for each data point in `X`. In the context of convex clustering, `U` can be thought of as a matrix where each row corresponds to the centroid of a cluster that a data point is assigned to.
- `F`: Represents the auxiliary variables that enforce the equality constraints necessary for the ADMM (Alternating Direction Method of Multipliers) algorithm to solve convex optimization problems. These constraints ensure that the differences between the centroids are properly accounted for in the optimization.

### Objective Function
- The data fidelity term is `0.5 * cp.sum_squares(U - X)`, which measures the squared Euclidean distance between each data point and its corresponding centroid. The algorithm aims to minimize this term.
- The regularization term `gamma * cp.sum(cp.norm(F, axis=1))` penalizes the sum of the Euclidean norms of the differences between all pairs of centroids. The regularization parameter `gamma` controls this term, and a larger `gamma` encourages fewer clusters by increasing the penalty for having distinct centroids.

## Constraints and Optimization Problem

### Constraints
- The list comprehension `'[F[i*n + j, :] == U[i, :] - U[j, :] for i in range(n) for j in range(n)]'` creates pairwise constraints for every pair of data points. This enforces that the differences between centroids (stored in `F`) are equal to the actual differences between the `U` variables. This part is crucial for the sum-of-norms clustering and is a direct translation of the mathematical constraints found in a convex clustering formulation.

### Optimization Problem
- The optimization problem is defined with the expression `problem = cp.Problem(cp.Minimize(objective), constraints)`. It encapsulates the objective of minimizing the objective function subject to the given constraints.
- The problem is solved using `problem.solve(solver=cp.SCS)`, which utilizes the SCS (Split Conic Solver), suitable for large-scale convex optimization problems.

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
