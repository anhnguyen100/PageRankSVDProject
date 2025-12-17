# A Comparative Analysis of SVD and PageRank in High-Dimensional Manifolds

## Overview

This project compares the robustness of three centrality/ranking methods on image manifolds:
- **PageRank** (Graph-based approach)
- **SVD** (Principal Component Analysis-based approach)
- **Hybrid** (PageRank applied to SVD features)

The analysis tests how well each method maintains consistent rankings when data is corrupted with Gaussian noise, using Fashion MNIST and Classic MNIST datasets.

## Features

- **Intrinsic Dimension Estimation**: Uses Costa & Hero method to measure manifold complexity
- **Robustness Testing**: Evaluates ranking consistency under increasing noise levels (0.0 to 0.5)
- **Stability Testing**: Measures Jaccard index of top-k rankings across noise levels
- **Visualization**: 
  - Robustness comparison plot
  - Prototype grids showing top representatives for each method
  - Manifold centrality map

## Requirements

- Python 3.7+
- numpy
- matplotlib
- torchvision
- scikit-learn
- scipy

## Installation

1. Clone or download this repository
2. Install dependencies:
```bash
python -m pip install -r requirements.txt
```

## Usage

Run the complete analysis:
```bash
python main.py
```

### Output

The script will:
1. Load 1000 samples from Fashion MNIST and Classic MNIST
2. Estimate intrinsic dimensions for both datasets
3. Run robustness test with noise levels from 0.0 to 0.5
4. Run stability test measuring ranking consistency
5. Generate three visualization windows:
   - Robustness comparison line plot
   - PageRank prototype grid
   - SVD prototype grid
   - Manifold centrality map

### Progress Messages

The script displays detailed progress for each step:
- [STEP 1/6] Loading datasets
- [STEP 2/6] Estimating intrinsic dimensions
- [STEP 3/6] Running robustness test
- [STEP 4/6] Running stability test
- [STEP 5/6] Generating visualizations
- [STEP 6/6] Analysis complete

## Configuration

Edit the main section in `main.py` to customize:
- **SAMPLES**: Number of training samples (default: 1000)
- **CLASS_ID**: Target class for analysis (default: 0)

### Sample Size Recommendations

- **500 samples**: Quick test on limited memory
- **1000 samples**: Balanced performance (recommended)
- **2000 samples**: More samples for better statistical results (requires more memory)

## Key Findings

The analysis helps determine:
- Which method is most robust to noise
- How manifold complexity affects ranking stability
- Whether hybrid approaches outperform single methods
- Visual representations of what each method considers "central"

## Project Structure

```
.
├── main.py              # Main analysis script
├── requirements.txt     # Python dependencies
├── README.md            # This file
└── data/                # Dataset directory (auto-created)
    ├── FashionMNIST/
    └── MNIST/
```

## References

- Costa, J. A., & Hero, A. O. (2004). Geodesic entropic graphs for dimension and entropy estimation in manifolds. IEEE Transactions on Signal Processing.
- Page, L., Brin, S., Motwani, R., & Winograd, T. (1999). The PageRank Citation Ranking: Bringing Order to the Web.
- Turk, M., & Pentland, A. (1991). Eigenfaces for recognition. Journal of Cognitive Neuroscience, 3(1), 71–86. 

## Notes

- First run will download datasets (may take a few minutes)
- Computation time depends on sample size and system memory
- Plot windows will appear sequentially; close each to continue
- Results are deterministic (seed set to 42)

## Authors

Cami Loyola
Helen Martinez
Anh Nguyen
