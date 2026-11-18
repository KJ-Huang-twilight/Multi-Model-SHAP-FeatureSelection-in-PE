"""
Intersect the top-N genes from two consensus rankings and re-rank by average rank.

Purpose
-------
The previous step produced two independent consensus rankings of genes:
  1. `gene_average_ranks_model.xlsx` — consensus ranking built from each
     model's *native* feature importance (e.g., `feature_importances_`, `coef_`).
  2. `gene_average_ranks_shap.xlsx`  — consensus ranking built from SHAP values.

This script takes the top-N genes (default N=100) from each ranking, finds the
genes that appear in *both* top-N lists (the intersection), and re-ranks those
common genes by their average within-list position (0-indexed). Genes that
rank highly in both the native-importance and SHAP-importance consensus lists
receive the smallest average position and are therefore the strongest
candidates for priority/robustness in downstream analysis.

Output: `common_top_genes_with_avg_rank.xlsx` with columns
    ["Common Genes", "Average Rank"].

Configuration
-------------
Change INPUT_DIR, FILE_NAMES, TOP_N and OUTPUT_FILE_PATH to match your setup.

Requirements
------------
pandas, openpyxl
"""

import os
import pandas as pd


def main():
    # -----------------------------------------------------------------------
    # Configuration: change these paths / parameters to your own setup.
    # -----------------------------------------------------------------------
    INPUT_DIR = "results"                              # directory of consensus ranking files
    FILE_NAMES = [                                     # the two consensus rankings to intersect
        "gene_average_ranks_shap.xlsx",
        "gene_average_ranks_model.xlsx",
    ]
    TOP_N = 100                                        # keep top-N genes from each ranking
    OUTPUT_FILE_PATH = os.path.join(INPUT_DIR, "common_top_genes_with_avg_rank.xlsx")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE_PATH), exist_ok=True)

    # Resolve file paths
    file_paths = [os.path.join(INPUT_DIR, f) for f in FILE_NAMES]

    # -----------------------------------------------------------------------
    # Load the top-N genes (first column, skipping the header) from each file
    # -----------------------------------------------------------------------
    dfs = [
        pd.read_excel(file_path).iloc[:TOP_N, 0].tolist()
        for file_path in file_paths
    ]

    # -----------------------------------------------------------------------
    # Record the 0-indexed position of each gene within each ranking
    # -----------------------------------------------------------------------
    gene_positions = {}
    for i, gene_list in enumerate(dfs):
        for rank, gene in enumerate(gene_list):
            if gene not in gene_positions:
                gene_positions[gene] = [None, None]
            gene_positions[gene][i] = rank

    # Keep only genes present in *both* top-N lists
    common_genes_with_positions = {
        gene: positions
        for gene, positions in gene_positions.items()
        if all(pos is not None for pos in positions)
    }

    # Average position across the two rankings
    common_genes_with_avg_rank = {
        gene: sum(positions) / len(positions)
        for gene, positions in common_genes_with_positions.items()
    }

    # Sort ascending by average position (smaller = more consistently top-ranked)
    sorted_common_genes_by_avg_rank = sorted(
        common_genes_with_avg_rank.items(), key=lambda item: item[1]
    )

    # Split into names and average ranks
    sorted_common_genes_names = [gene for gene, _ in sorted_common_genes_by_avg_rank]
    sorted_avg_ranks = [avg_rank for _, avg_rank in sorted_common_genes_by_avg_rank]

    # -----------------------------------------------------------------------
    # Save to Excel
    # -----------------------------------------------------------------------
    result_df = pd.DataFrame(
        {
            "Common Genes": sorted_common_genes_names,
            "Average Rank": sorted_avg_ranks,
        }
    )
    result_df.to_excel(OUTPUT_FILE_PATH, index=False)

    print(
        f"{len(sorted_common_genes_names)} common genes saved to {OUTPUT_FILE_PATH}"
    )


if __name__ == "__main__":
    main()