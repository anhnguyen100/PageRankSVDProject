# PageRank & SVD Prototypes on Fashion-MNIST

This project explores how **graph-based centrality (PageRank)** and **low-rank linear structure (Truncated SVD)** can be used to:

- Select **prototype** images for each Fashion-MNIST class  
- Quantitatively compare how *central* those prototypes are  
- Compare two simple **compression** methods for images  

All experiments are done on a subset of the **Fashion-MNIST** dataset.

---

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [How It Works](#how-it-works)
  - [1. Data Loading](#1-data-loading)
  - [2. k-NN Similarity Graph](#2-k-nn-similarity-graph)
  - [3. PageRank](#3-pagerank)
  - [4. Truncated SVD](#4-truncated-svd)
  - [5. Prototype Selection](#5-prototype-selection)
  - [6. Centrality Error](#6-centrality-error)
  - [7. Compression Comparison](#7-compression-comparison)
- [Running the Code](#running-the-code)
- [Changing Parameters](#changing-parameters)
- [Project Structure](#project-structure)
- [Acknowledgments](#acknowledgments)

---

## Overview

The main script:

1. Loads a subset of **Fashion-MNIST**.
2. Normalizes the images and builds a **k-nearest neighbor (k-NN) graph** using cosine similarity.
3. Runs **PageRank** on that graph to get a centrality score for each image.
4. Runs **Truncated SVD** to obtain a low-dimensional representation.
5. Selects **prototype images** for a given class in two ways:
   - Top PageRank images within that class.
   - Images closest to the class centroid in SVD space.
6. Computes a **centrality error** to evaluate how representative each prototype set is.
7. Compares **SVD-based reconstruction** vs. a **sparse pixel mask** reconstruction.

---

## Requirements

Tested with Python 3.10+, but Python 3.8+ should work.

Install dependencies:

    pip install numpy torchvision torch scikit-learn scipy matplotlib

---

## How It Works

### 1. Data Loading

`load_fashion_mnist(n_samples=1000)`

- Downloads the Fashion-MNIST training set via `torchvision.datasets.FashionMNIST`.
- Takes the first `n_samples` images.
- Flattens each `(1, 28, 28)` image into a 784-dimensional vector.
- Returns:
  - `X`: shape `(n_samples, 784)` — image pixels.
  - `y`: shape `(n_samples,)` — labels in `{0, …, 9}`.

---

### 2. k-NN Similarity Graph

`build_knn_graph(X_norm, k=10)`

- Uses `sklearn.neighbors.NearestNeighbors` with **cosine distance**.
- For each image, finds `k` nearest neighbors (excluding itself).
- Converts cosine distance `d` into similarity `sim = 1 - d`.
- Constructs a sparse adjacency matrix `A` where `A[i, j]` stores the similarity from image `i` to `j`.

This gives a sparse similarity graph over all images.

---

### 3. PageRank

`run_pagerank(A, alpha=0.85, tol=1e-6, max_iter=100)`

- Row-normalizes `A` to obtain a transition matrix `P`.
- Runs power iteration with damping factor `alpha`:

  r_new = alpha * P^T * r + (1 - alpha) * v

  where `v` is the uniform teleportation vector.

- Iterates until the L1 difference between `r_new` and `r` is below `tol` or `max_iter` is reached.
- Returns a PageRank score `r[i]` for every image.

These scores are used to:
- Find globally central images.
- Pick central images **within** each class.

---

### 4. Truncated SVD

`run_svd(X_norm, n_components=50)`

- Applies `sklearn.decomposition.TruncatedSVD` to the normalized data.
- Returns:
  - `svd_model`: the fitted SVD model (with `components_`).
  - `Z`: low-dimensional representation of all images, shape `(n_samples, n_components)`.

Each row `Z[i]` is the compressed representation of image `i`.

---

### 5. Prototype Selection

#### PageRank Prototypes (Within a Class)

`top_pagerank_for_class(y, scores, label, k=10)`

- Filters indices where `y == label`.
- Sorts those indices by descending PageRank score.
- Returns the top `k` PageRank-based prototypes for that class.

#### SVD Prototypes (Within a Class)

`svd_prototypes_for_class(y, Z, label, r=20, k=10)`

- Selects images with label `label`.
- Uses the top `r` SVD components for that class: `Z_c = Z[class_indices, :r]`.
- Computes the class centroid in this r-dimensional space.
- Finds the `k` images closest to the centroid in Euclidean distance.
- Returns the indices of these SVD-based prototypes.

---

### 6. Centrality Error

`centrality_error(X_norm, y, prototype_indices, label)`

Measures how well a prototype set represents its class:

1. Selects all images in the class `label`.
2. Computes pairwise Euclidean distances between class images and prototypes.
3. For each class image, finds the distance to its **closest** prototype.
4. Returns the **average** of these minimum distances.

Lower centrality error ⇒ prototypes are more representative / central.

This metric is used to compare PageRank vs. SVD prototypes.

---

### 7. Compression Comparison

#### SVD Reconstruction

`reconstruct_svd_image(X_norm, svd_model, idx, r=20)`

- Takes image `idx` from `X_norm`.
- Projects it into SVD space using `svd_model.transform`.
- Keeps only the top `r` components.
- Reconstructs the image using `V_r = svd_model.components_[:r, :]`.
- Returns:
  - Original normalized image (reshaped to `28×28`).
  - SVD reconstruction (also `28×28`).

#### Sparse Mask Reconstruction

`reconstruct_sparse_mask(X, idx, keep_ratio=0.2)`

- Uses the original (unnormalized) image `X[idx]`.
- Computes absolute pixel intensities.
- Finds a threshold so that only the top `keep_ratio` fraction of pixels (by intensity) are kept.
- Zeros out all other pixels.
- Returns:
  - Original image (`28×28`).
  - Masked (sparse) reconstruction (`28×28`).

Both reconstructions are compared with mean squared error (MSE).

---

## Running the Code

Assuming the script is saved as `main.py`:

    python main.py

You will see:

- Console logs:
  - Data loading progress.
  - k-NN graph construction.
  - PageRank iterations and convergence.
  - SVD completion.
  - Centrality error comparison for the chosen class (default: `0`).
  - MSE for SVD vs. sparse mask reconstruction on a selected image.
- Matplotlib windows displaying:
  - Top PageRank prototypes for the target class.
  - Top SVD prototypes for the target class.
  - Original vs. SVD vs. sparse-mask reconstruction for the chosen test image.

---

## Changing Parameters

You can tweak key parameters directly in `main()`:

- **Number of samples**

      n_samples = 1000

- **k in k-NN graph**

      A = build_knn_graph(X_norm, k=10)

- **PageRank damping factor**

      pagerank_scores = run_pagerank(A, alpha=0.85)

- **Number of SVD components**

      svd_model, Z = run_svd(X_norm, n_components=50)

- **SVD prototype parameters**

      svd_idx = svd_prototypes_for_class(y, Z, label=target_class, r=20, k=10)

- **Target class**

      target_class = 0

- **Sparse mask keep ratio**

      orig_mask, recon_mask = reconstruct_sparse_mask(X, test_index, keep_ratio=0.20)

Adjust these to explore different dataset sizes, neighborhood sizes, prototype counts, and compression strengths.

---

## Project Structure

Everything is currently in one file:

- `main.py`  
  Contains:
  - Data loading (`load_fashion_mnist`)
  - Graph construction (`build_knn_graph`)
  - PageRank (`run_pagerank`)
  - SVD (`run_svd`)
  - Prototype selection (`top_pagerank_for_class`, `svd_prototypes_for_class`)
  - Centrality error computation (`centrality_error`)
  - Reconstruction / compression comparison (`reconstruct_svd_image`, `reconstruct_sparse_mask`)
  - Plotting helpers (`show_images`)
  - `main()` pipeline

---

## Acknowledgments

- **Fashion-MNIST**: Zalando Research  
- **PageRank**: Sergey Brin & Larry Page, *The Anatomy of a Large-Scale Hypertextual Web Search Engine*  
- **Truncated SVD & k-NN**: Implemented via scikit-learn  
- **Dataset & transforms**: via torchvision