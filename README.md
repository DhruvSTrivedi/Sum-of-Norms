# Enhanced Convex Clustering via Sum-of-Norms Methodology

## Introduction

In the realm of machine learning, particularly under unsupervised learning paradigms, convex optimization stands out as a pivotal area of study. This method is characterized by its objective function's convex nature, simplifying the optimization process by guaranteeing that each local minimum is also a global minimum. This inherent property renders convex optimization problems more manageable compared to their non-convex counterparts.

Unsupervised machine learning extensively employs convex optimization to optimally group data points by minimizing a specific cost function. Clustering algorithms such as k-means and sum-of-norms clustering are prime examples, leveraging convex optimization for efficient data segmentation. Contrary to supervised learning algorithms, which utilize labeled data for training, unsupervised algorithms like clustering deduce the inherent structure within a dataset without prior label information.

Explore our detailed findings and methodologies in our Jupyter Notebook:  
[Jupyter Notebook](Kmeans vs. Sum of Norms.ipynb)

This research is grounded in the work of **[Dr. Stephen Vavasis](https://uwaterloo.ca/combinatorics-and-optimization/contacts/stephen-vavasis)**, and for an in-depth understanding, please refer to his publication:  
[Certifying clusters from sum-of-norms clustering]

...

[The rest of your content continues in Markdown format]


  This research is grounded in the work of **[Dr. Stephen Vavasis](https://uwaterloo.ca/combinatorics-and-optimization/contacts/stephen-vavasis)**, and for an in-depth understanding, please refer to his publication:  
  [Certifying clusters from sum-of-norms clustering]

## Convex Clustering: Variables and Objective Function

### Core Variables
- `U`: Denotes the cluster centroids corresponding to each data point in `X`. In convex clustering, `U` is conceptualized as a matrix where each row aligns with the centroid of a cluster assigned to a data point.
- `F`: Symbolizes auxiliary variables essential for enforcing the equality constraints in the ADMM (Alternating Direction Method of Multipliers) algorithm for solving convex optimization problems. These constraints are vital for appropriately accounting for differences between centroids in the optimization process.

### Objective Function
- Data Fidelity Term: `0.5 * cp.sum_squares(U - X)`, quantifying the squared Euclidean distance between data points and their respective centroids. The goal is to minimize this term.
- Regularization Term: `gamma * cp.sum(cp.norm(F, axis=1))`, imposing penalties on the sum of the Euclidean norms of centroid differences. The `gamma` parameter regulates this term, where a higher `gamma` value fosters fewer clusters by intensifying the penalty for distinct centroids.

## Constraints and Optimization Strategy

### Constraints
- The expression `'[F[i*n + j, :] == U[i, :] - U[j, :] for i in range(n) for j in range(n)]'` establishes pairwise constraints for each data point pair. This ensures that the centroid differences (stored in `F`) match the actual differences in the `U` variables, which is crucial for sum-of-norms clustering.

### Optimization Problem
- Defined as `problem = cp.Problem(cp.Minimize(objective), constraints)`, this encapsulates the goal of minimizing the objective function under the specified constraints.
- The problem is tackled using `problem.solve(solver=cp.SCS)`, employing the SCS (Split Conic Solver), adept for large-scale convex optimization challenges.

Each point in our visual representation corresponds to a data instance, with color coding indicating cluster assignments.

## Theoretical Insights

Sum-of-norms clustering stands apart from K-means in several critical areas, including the objective function, cluster configurations, cluster count determination, outlier resilience, computational complexity, and interpretability.

## Comparative Analysis
| Aspect | K-means Clustering | Sum-of-Norms Clustering |
|--------|--------------------|-------------------------|
| **Objective Function** | Seeks to divide data into `K` clusters, minimizing the within-cluster sum of squares. | Aims to minimize a mix of squared differences and a sparsity-promoting regularization term for centroid variations. |
| **Cluster Configuration** | Presumes spherical clusters of similar sizes, a potential limitation. | Flexible regarding shape and size, capable of identifying clusters with irregular contours and hierarchical structures. |
| **Cluster Count Determination** | Requires pre-setting the number of clusters (`K`). | Determines cluster count based on data and `gamma`; a higher `gamma` results in fewer clusters. |
| **Outlier Resilience** | Potentially sensitive to outliers, significantly influenced by mean cluster values. | Enhanced robustness against outliers due to the regularization term, which promotes similar cluster assignments. |
| **Computation** | Typically faster, scaling efficiently with larger datasets. | Computationally demanding, involving convex optimization, especially for larger datasets. |
| **Interpretability** | Straightforward but may oversimplify clustering dynamics. | Offers a nuanced perspective, uncovering complex clustering patterns possibly overlooked by K-means. |

## Conclusion
Sum-of-norms clustering unveils intricate data patterns, a capability that simpler algorithms like K-means might overlook. This approach is particularly effective for datasets featuring complex groupings. Our project demonstrates how sum-of-norms clustering can yield distinct data groupings compared to K-means, with potential variances in cluster numbers, shapes, and resilience to outliers. The accompanying visual representations highlight these differences, offering a comparative view of cluster assignments and structures.

---
