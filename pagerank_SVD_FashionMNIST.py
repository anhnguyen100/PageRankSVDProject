import numpy as np
from torchvision import datasets, transforms
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import pairwise_distances
from scipy import sparse
import matplotlib.pyplot as plt


def load_fashion_mnist(n_samples=1000):
    """Load a subset of Fashion MNIST and return X, y."""
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_ds = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=transform
    )

    X_list = []
    y_list = []

    for i in range(n_samples):
        img, label = train_ds[i]
        # img has shape (1, 28, 28)
        X_list.append(img.numpy().reshape(-1))  # flatten to 784
        y_list.append(label)

    X = np.stack(X_list, axis=0)
    y = np.array(y_list)
    return X, y


def build_knn_graph(X_norm, k=10):
    """Build a k nearest neighbor graph using cosine distance. Return adjacency matrix A."""
    n_samples = X_norm.shape[0]

    nn = NearestNeighbors(
        n_neighbors=k + 1,   
        metric="cosine"
    ).fit(X_norm)

    distances, indices = nn.kneighbors(X_norm)

    rows = []
    cols = []
    data = []

    for i in range(n_samples):
        
        for j_idx, d in zip(indices[i, 1:], distances[i, 1:]):
            sim = 1.0 - d          # cosine similarity
            if sim > 0:
                rows.append(i)
                cols.append(j_idx)
                data.append(sim)

    A = sparse.csr_matrix((data, (rows, cols)),
                          shape=(n_samples, n_samples))
    return A


def run_pagerank(A, alpha=0.85, tol=1e-6, max_iter=100):
    """Compute PageRank scores on adjacency matrix A."""
    n = A.shape[0]

    # row normalize A to get transition matrix P
    row_sums = np.array(A.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0
    D_inv = sparse.diags(1.0 / row_sums)
    P = D_inv @ A

    r = np.ones(n) / n
    v = np.ones(n) / n

    for it in range(max_iter):
        r_new = alpha * P.T.dot(r) + (1.0 - alpha) * v
        diff = np.linalg.norm(r_new - r, 1)
        print(f"Iteration {it + 1}: L1 diff = {diff:.2e}")
        r = r_new
        if diff < tol:
            break

    return r


def run_svd(X_norm, n_components=50):
    """Run TruncatedSVD on normalized data. Return model and low dimensional features Z."""
    svd = TruncatedSVD(n_components=n_components, random_state=0)
    Z = svd.fit_transform(X_norm)
    return svd, Z


def svd_prototypes_for_class(y, Z, label, r=20, k=10):
    """Return indices of SVD based prototypes for a given class."""
    mask = (y == label)
    idx = np.where(mask)[0]
    Z_c = Z[idx, :r]              
    centroid = Z_c.mean(axis=0)
    dists = np.linalg.norm(Z_c - centroid, axis=1)
    order = np.argsort(dists)
    return idx[order[:k]]


def top_pagerank_for_class(y, scores, label, k=10):
    """Return indices of top PageRank images for a given class."""
    mask = (y == label)
    idx = np.where(mask)[0]
    idx_sorted = idx[np.argsort(-scores[idx])]
    return idx_sorted[:k]


def show_images(indices, X, title):
    """Simple grid plot of images given their indices."""
    n = len(indices)
    cols = min(n, 10)
    rows = (n + cols - 1) // cols

    plt.figure(figsize=(1.5 * cols, 1.5 * rows))
    for i, idx in enumerate(indices):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(X[idx].reshape(28, 28), cmap="gray")
        plt.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def reconstruct_svd_image(X_norm, svd_model, idx, r=20):
    """Reconstruct a single image using top r SVD components."""
    x = X_norm[idx:idx+1]         #shape (1, 784)
    Z = svd_model.transform(x)[:, :r]
    V_r = svd_model.components_[:r, :]
    x_hat = Z.dot(V_r)
    return x.reshape(28, 28), x_hat.reshape(28, 28)


def reconstruct_sparse_mask(X, idx, keep_ratio=0.2):
    """
    Reconstruct a single image by keeping only the top keep_ratio fraction
    of its highest intensity pixels.
    """
    img_flat = X[idx]                 #shape (784,)
    scores = np.abs(img_flat)         #pixel importance = intensity

    # threshold so that only top keep_ratio fraction of pixels are kept
    threshold = np.quantile(scores, 1 - keep_ratio)
    mask_flat = (scores >= threshold).astype(float)

    img = img_flat.reshape(28, 28)
    mask = mask_flat.reshape(28, 28)

    return img, img * mask


def centrality_error(X_norm, y, prototype_indices, label):
    """
    Compute how well the prototypes represent the class.

    Average distance from each image in the class to its nearest prototype.
    Lower average distance means more central prototypes.
    """
    class_indices = np.where(y == label)[0]
    X_class = X_norm[class_indices]
    X_protos = X_norm[prototype_indices]

    dists = pairwise_distances(X_class, X_protos, metric="euclidean")
    min_dists = np.min(dists, axis=1)
    return float(np.mean(min_dists))


def main():
    # 1. load data
    n_samples = 1000
    print("Loading Fashion MNIST...")
    X, y = load_fashion_mnist(n_samples=n_samples)
    print("Done.")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # 2. normalize features
    print("Normalizing...")
    X_norm = normalize(X, norm="l2")

    # 3. build k nearest neighbor graph
    print("Building k-NN graph...")
    A = build_knn_graph(X_norm, k=10)
    print("Adjacency matrix shape:", A.shape)
    print("Number of edges:", A.nnz)

    # 4. run PageRank
    print("Running PageRank...")
    pagerank_scores = run_pagerank(A, alpha=0.85)
    print("PageRank complete.")
    top_indices = np.argsort(-pagerank_scores)[:10]
    print("Top 10 PageRank indices:", top_indices)
    print("Labels of those images:", y[top_indices])

    # 5. run SVD
    print("Running SVD...")
    svd_model, Z = run_svd(X_norm, n_components=50)
    print("SVD complete. Z shape:", Z.shape)

    # 6. compare PageRank vs SVD prototypes for one class
    target_class = 0
    print(f"Showing prototypes for class {target_class}.")

    pr_idx = top_pagerank_for_class(y, pagerank_scores, label=target_class, k=10)
    svd_idx = svd_prototypes_for_class(y, Z, label=target_class, r=20, k=10)

    # 6.5 centrality error comparison
    pr_err = centrality_error(X_norm, y, pr_idx, label=target_class)
    svd_err = centrality_error(X_norm, y, svd_idx, label=target_class)

    print("\nCENTRALITY ERROR COMPARISON")
    print(f"PageRank prototypes (class {target_class}): {pr_err:.4f}")
    print(f"SVD prototypes (class {target_class}): {svd_err:.4f}")
    if pr_err < svd_err:
        print("PageRank produces more central prototypes.")
    else:
        print("SVD produces more central prototypes.")

    # 7. plot prototypes
    show_images(pr_idx, X, f"Top PageRank prototypes for class {target_class}")
    show_images(svd_idx, X, f"Top SVD prototypes for class {target_class}")

    # 8. compression example on the most central PageRank image
    test_index = int(top_indices[0])
    print(f"\nReconstructing image {test_index} using SVD and sparse mask...")

    # SVD reconstruction
    orig_svd, recon_svd = reconstruct_svd_image(X_norm, svd_model, test_index, r=20)

    # sparse mask reconstruction on the same image
    orig_mask, recon_mask = reconstruct_sparse_mask(X, test_index, keep_ratio=0.20)

    # compute MSE for both reconstructions
    mse_svd = np.mean((orig_svd - recon_svd) ** 2)
    mse_mask = np.mean((orig_mask - recon_mask) ** 2)
    print(f"MSE of SVD reconstruction: {mse_svd:.6f}")
    print(f"MSE of sparse mask reconstruction: {mse_mask:.6f}")

    # plot both
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(orig_svd, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("SVD Reconstruction")
    plt.imshow(recon_svd, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Sparse Mask Reconstruction")
    plt.imshow(recon_mask, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
