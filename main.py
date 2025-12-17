import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.metrics import pairwise_distances
from scipy import sparse
from scipy.stats import linregress


# 1. DATA LOADING & PREPROCESSING

def load_data(dataset_name="fashion", n_samples=2000):
    """
    Load a subset of Fashion MNIST or Classic MNIST.
    """
    transform = transforms.Compose([transforms.ToTensor()])
    
    if dataset_name == "fashion":
        ds = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
        print(f"Loading Fashion MNIST ({n_samples} samples)...")
    else:
        ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        print(f"Loading Classic MNIST ({n_samples} samples)...")

    # Deterministic sampling for consistent results
    np.random.seed(42)
    indices = np.random.choice(len(ds), n_samples, replace=False)
    
    X_list = []
    y_list = []
    
    for i in indices:
        img, label = ds[i]
        X_list.append(img.numpy().reshape(-1))
        y_list.append(label)

    X = np.stack(X_list, axis=0)
    y = np.array(y_list)
    return X, y


# 2. INTRINSIC DIMENSION (Costa & Hero Method)

def compute_k_nn_length(X, k=10, gamma=1.0):
    """Calculates the total edge length of the k-NN graph."""
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    distances, _ = nn.kneighbors(X)
    # Sum of distances raised to power gamma (exclude self-distance at col 0)
    total_length = np.sum(distances[:, 1:] ** gamma)
    return total_length

def estimate_intrinsic_dimension(X, k=5, gamma=1.0):
    """
    Estimates intrinsic dimension using the Growth Rate of the k-NN graph.
    Reference: Costa & Hero (2004).
    """
    N = X.shape[0]
    # Define subsample sizes (log-spaced)
    p_values = np.unique(np.logspace(np.log10(50), np.log10(N), num=10, dtype=int))
    p_values = p_values[p_values < N] 
    
    avg_lengths = []
    n_bootstraps = 5
    
    for p in p_values:
        lengths = []
        for _ in range(n_bootstraps):
            indices = np.random.choice(N, p, replace=False)
            X_sub = X[indices]
            L = compute_k_nn_length(X_sub, k, gamma)
            lengths.append(L)
        avg_lengths.append(np.mean(lengths))
        
    # Linear Regression: log(L) vs log(n)
    log_n = np.log(p_values)
    log_L = np.log(avg_lengths)
    
    slope, _, _, _, _ = linregress(log_n, log_L)
    
    # Formula: m = gamma / (1 - slope)
    if slope >= 0.99:
        return 0.0 # Undefined/Error case
    else:
        return gamma / (1.0 - slope)


# 3. RANKING ALGORITHMS

def build_graph_and_pagerank(X, k=10, metric="euclidean", sigma=None):
    """
    Builds k-NN graph and runs PageRank.
    Uses Gaussian Kernel when metric='euclidean'.
    """
    n_samples = X.shape[0]
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(X)
    distances, indices = nn.kneighbors(X)

    rows, cols, data = [], [], []
    
    # Heuristic for sigma: average distance to k-th neighbor
    if sigma is None and metric == "euclidean":
        sigma = np.mean(distances[:, -1])

    for i in range(n_samples):
        for j_idx, d in zip(indices[i, 1:], distances[i, 1:]):
            if metric == "cosine":
                sim = max(0, 1.0 - d)
            else:
                # Gaussian Kernel
                sim = np.exp(-(d**2) / (2 * sigma**2))
            
            if sim > 1e-10:
                rows.append(i)
                cols.append(j_idx)
                data.append(sim)

    A = sparse.csr_matrix((data, (rows, cols)), shape=(n_samples, n_samples))
    
    # PageRank Power Iteration
    row_sums = np.array(A.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0
    D_inv = sparse.diags(1.0 / row_sums)
    P = D_inv @ A
    
    r = np.ones(n_samples) / n_samples
    for _ in range(50): 
        r = 0.85 * P.T.dot(r) + 0.15 * (1.0/n_samples)
        
    return r

def run_svd_ranking(X, n_components=50):
    """SVD Ranking based on centrality in reduced space."""
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    Z = svd.fit_transform(X)
    
    mean_vec = np.mean(Z, axis=0)
    dists = np.linalg.norm(Z - mean_vec, axis=1)
    
    # Score = Closeness
    scores = 1.0 / (1.0 + dists) 
    return scores, Z


# 4. EVALUATION & METRICS

def calculate_centrality_error(X, y, scores, label, k=5):
    """Metric: Average distance of class members to nearest prototype."""
    class_idx = np.where(y == label)[0]
    class_scores = scores[class_idx]
    
    best_local_indices = np.argsort(-class_scores)[:k]
    prototype_indices = class_idx[best_local_indices]
    
    X_class = X[class_idx]
    X_protos = X[prototype_indices]
    
    dists = pairwise_distances(X_class, X_protos, metric="euclidean")
    min_dists = np.min(dists, axis=1)
    
    return np.mean(min_dists)

def jaccard_similarity(list1, list2):
    """Calculates stability via Jaccard similarity."""
    s1 = set(list1)
    s2 = set(list2)
    if len(s1) == 0 or len(s2) == 0: return 0.0
    return len(s1.intersection(s2)) / len(s1.union(s2))


# 5. EXPERIMENTS: ROBUSTNESS & STABILITY TESTS

def run_robustness_test(X, y, target_class=0):
    """Tracks Centrality Error as noise increases."""
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    results = {'pr': [], 'svd': [], 'hybrid': []}
    
    print(f"\n--- Running Robustness Test (Target Class: {target_class}) ---")
    
    for sigma in noise_levels:
        noise = np.random.normal(0, sigma, X.shape)
        X_noisy = normalize(X + noise, norm="l2")
        
        # 1. PageRank (Gaussian)
        pr_scores = build_graph_and_pagerank(X_noisy, metric="euclidean")
        err_pr = calculate_centrality_error(X_noisy, y, pr_scores, target_class)
        results['pr'].append(err_pr)
        
        # 2. SVD
        svd_scores, Z = run_svd_ranking(X_noisy)
        err_svd = calculate_centrality_error(X_noisy, y, svd_scores, target_class)
        results['svd'].append(err_svd)
        
        # 3. Hybrid
        hybrid_scores = build_graph_and_pagerank(Z, metric="euclidean")
        err_hyb = calculate_centrality_error(X_noisy, y, hybrid_scores, target_class)
        results['hybrid'].append(err_hyb)
        
        print(f"Noise {sigma:.1f} | PR: {err_pr:.3f} | SVD: {err_svd:.3f} | Hyb: {err_hyb:.3f}")
        
    return noise_levels, results

def run_stability_test(X, y, target_class=0, k=10):
    """Tracks Ranking Stability (Jaccard) as noise increases."""
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    print(f"\n--- Running Stability Test (Jaccard Index) ---")
    
    X_clean = normalize(X, norm="l2")
    
    # Baselines
    pr_base = build_graph_and_pagerank(X_clean, metric="euclidean")
    pr_idx_base = np.argsort(-pr_base[y == target_class])[:k]
    
    svd_base, _ = run_svd_ranking(X_clean)
    svd_idx_base = np.argsort(-svd_base[y == target_class])[:k]
    
    _, Z_clean = run_svd_ranking(X_clean)
    hyb_base = build_graph_and_pagerank(Z_clean, metric="euclidean")
    hyb_idx_base = np.argsort(-hyb_base[y == target_class])[:k]

    for sigma in noise_levels:
        noise = np.random.normal(0, sigma, X.shape)
        X_noisy = normalize(X + noise, norm="l2")
        
        # PR Stability
        pr_new = build_graph_and_pagerank(X_noisy, metric="euclidean")
        pr_idx_new = np.argsort(-pr_new[y == target_class])[:k]
        sim_pr = jaccard_similarity(pr_idx_base, pr_idx_new)
        
        # SVD Stability
        svd_new, Z_new = run_svd_ranking(X_noisy)
        svd_idx_new = np.argsort(-svd_new[y == target_class])[:k]
        sim_svd = jaccard_similarity(svd_idx_base, svd_idx_new)
        
        # Hybrid Stability
        hyb_new = build_graph_and_pagerank(Z_new, metric="euclidean")
        hyb_idx_new = np.argsort(-hyb_new[y == target_class])[:k]
        sim_hyb = jaccard_similarity(hyb_idx_base, hyb_idx_new)
        
        print(f"Noise {sigma:.1f} | PR Stab: {sim_pr:.2f} | SVD Stab: {sim_svd:.2f} | Hyb Stab: {sim_hyb:.2f}")


# 6. VISUALIZATIONS

def visualize_prototypes(X, y, scores, label, method_name, k=5):
    """Display top-k images selected by a method."""
    class_idx = np.where(y == label)[0]
    class_scores = scores[class_idx]
    
    best_local_indices = np.argsort(-class_scores)[:k]
    original_indices = class_idx[best_local_indices]
    
    plt.figure(figsize=(10, 3))
    for i, idx in enumerate(original_indices):
        plt.subplot(1, k, i + 1)
        img = X[idx].reshape(28, 28)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        if i == 0: plt.title(f"{method_name}\nRank {i+1}")
        else: plt.title(f"Rank {i+1}")
    plt.tight_layout()
    plt.show()

def plot_manifold_centrality(X, y, pr_scores, svd_scores, label):
    """2D PCA Map showing where prototypes sit in the cloud."""
    mask = (y == label)
    X_class = X[mask]
    
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_class)
    
    pr_top = np.argsort(-pr_scores[mask])[:5]
    svd_top = np.argsort(-svd_scores[mask])[:5]
    
    plt.figure(figsize=(8, 6))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c='lightgray', alpha=0.5, label='Class Members')
    plt.scatter(X_2d[pr_top, 0], X_2d[pr_top, 1], c='blue', s=100, marker='*', label='PageRank Top-5')
    plt.scatter(X_2d[svd_top, 0], X_2d[svd_top, 1], c='red', s=100, marker='X', label='SVD Top-5')
    
    plt.title(f"Manifold Map: Prototype Locations (Class {label})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


# 7. MAIN

if __name__ == "__main__":
    SAMPLES = 1000 
    CLASS_ID = 0
    
    print("=" * 70)
    print("STARTING ROBUSTNESS ANALYSIS FOR PAGERANK, SVD, AND HYBRID METHODS")
    print("=" * 70)
    
    print("\n[STEP 1/6] Loading datasets...")
    print(f"  Loading {SAMPLES} samples from Fashion MNIST...")
    X_f, y_f = load_data("fashion", SAMPLES)
    print(f"  Fashion MNIST loaded (shape: {X_f.shape})")
    
    print(f"  Loading {SAMPLES} samples from Classic MNIST...")
    X_m, y_m = load_data("mnist", SAMPLES)
    print(f"  Classic MNIST loaded (shape: {X_m.shape})")
    
    print("\n[STEP 2/6] Estimating Intrinsic Dimension (Costa & Hero Method)...")
    print("  Computing Fashion MNIST intrinsic dimension...")
    dim_f = estimate_intrinsic_dimension(X_f, k=5)
    print(f"  Fashion MNIST Intrinsic Dim: {dim_f:.2f}")
    
    print("  Computing Classic MNIST intrinsic dimension...")
    dim_m = estimate_intrinsic_dimension(X_m, k=5)
    print(f"  Classic MNIST Intrinsic Dim: {dim_m:.2f}")
    
    print("\n[STEP 3/6] Running Robustness Test (adding noise and comparing methods)...")
    print(f"  Testing with noise levels: 0.0, 0.1, 0.2, 0.3, 0.4, 0.5")
    print(f"  Target class: {CLASS_ID}")
    
    # Initial normalization
    X_f_norm = normalize(X_f, norm="l2")
    
    # 2. Robustness Test (Centrality Error)
    sigmas, res = run_robustness_test(X_f, y_f, target_class=CLASS_ID)
    print(f"  Robustness test completed!")
    
    # 3. Stability Test (Jaccard Index)
    print("\n[STEP 4/6] Running Stability Test (measuring ranking consistency)...")
    run_stability_test(X_f, y_f, target_class=CLASS_ID)
    print(f"  Stability test completed!")
    
    # 4. Generate Plots
    print("\n[STEP 5/6] Generating visualizations...")
    
    print("  Creating robustness line plot...")
    # Robustness Line Plot
    plt.figure(figsize=(10, 6))
    plt.plot(sigmas, res['pr'], 'o--', label='PageRank (Gaussian)', linewidth=2)
    plt.plot(sigmas, res['svd'], 's-', label='SVD', linewidth=2)
    plt.plot(sigmas, res['hybrid'], '^:', label='Hybrid', linewidth=2)
    plt.title(f"Robustness Analysis (Gaussian Kernel)")
    plt.xlabel("Gaussian Noise Level")
    plt.ylabel("Centrality Error (Lower is Better)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    print("  Robustness plot displayed!")
    
    print("  Generating prototype visualization grids...")
    # Prototype Grids
    pr_scores_vis = build_graph_and_pagerank(X_f_norm, metric="euclidean")
    svd_scores_vis, _ = run_svd_ranking(X_f_norm)
    
    visualize_prototypes(X_f, y_f, pr_scores_vis, CLASS_ID, "PageRank (Top 5)")
    visualize_prototypes(X_f, y_f, svd_scores_vis, CLASS_ID, "SVD (Top 5)")
    print("  Prototype grids displayed!")

    print("  Creating manifold centrality map...")
    # Manifold Map
    plot_manifold_centrality(X_f_norm, y_f, pr_scores_vis, svd_scores_vis, CLASS_ID)
    print("  Manifold map displayed!")
    
    print("\n[STEP 6/6] Analysis complete")
    print("=" * 70)
    print("ALL RESULTS GENERATED AND DISPLAYED SUCCESSFULLY")
    print("=" * 70)