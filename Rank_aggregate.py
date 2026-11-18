"""
Aggregate feature-importance rankings across multiple models into a single
consensus ranking.

Purpose
-------
For each model (e.g., Random Forest, XGBoost, LightGBM, SVM, MLP, etc.), the
pipeline produces a table of feature importances, either from the model's own
native importance (`feature_importances_` / `coef_`) or from SHAP values. Each
table is assumed to have columns [Feature, Importance] with features already
sorted in descending order of importance.

This script reads all such tables in a directory, assigns each gene/feature a
rank within its own model (1 = most important), filters out zero-importance
features, and finally computes the *average rank* of each gene across models.
Genes consistently ranked near the top across many models receive the
smallest average rank and therefore appear at the top of the consensus
ranking. Genes present in only a subset of models are still scored (their
average rank is computed over the models where they appear), and the output
table reports both the average rank and the number of models in which each
gene was found.

Configuration
-------------
Change INPUT_DIR and OUTPUT_FILE_PATH to point at your own directories.

Requirements
------------
pandas, openpyxl
"""

import os
import pandas as pd


def ensure_gene_names_are_strings(names):
    """Ensure that the given names are strings and strip surrounding whitespace."""
    return [str(name).strip() for name in names]


def load_gene_ranks(file_path):
    """Load a [Feature, Importance] table and return the list of features
    with non-zero importance, preserving the original row order (assumed to
    be already sorted by importance)."""
    try:
        # Read the Excel file, forcing the first column to be strings
        df = pd.read_excel(file_path, engine="openpyxl", dtype={0: "str"})

        # Assume the first column holds feature names, the second importance scores
        genes = df.iloc[:, 0]
        importance_scores = df.iloc[:, 1]

        # Clean and standardize feature names
        cleaned_genes = ensure_gene_names_are_strings(genes)

        # Drop features with zero importance
        filtered_genes = [
            gene
            for gene, score in zip(cleaned_genes, importance_scores)
            if score != 0
        ]

        print(f"Loaded {len(filtered_genes)} non-zero importance genes from {file_path}")
        return filtered_genes

    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []


def main():
    # -----------------------------------------------------------------------
    # Configuration: change these paths to your own directories.
    # -----------------------------------------------------------------------
    INPUT_DIR = "data/model_importance"       # directory containing per-model .xlsx files
    OUTPUT_FILE_PATH = "results/gene_average_ranks.xlsx"  # consensus ranking output

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE_PATH), exist_ok=True)

    # Collect all .xlsx files in the input directory
    all_files = [
        os.path.join(INPUT_DIR, f)
        for f in os.listdir(INPUT_DIR)
        if f.endswith(".xlsx")
    ]
    print("Found files:", all_files)

    # Dictionaries to accumulate rank sums and per-gene counts
    gene_ranks_sum = {}
    gene_count = {}

    for file in all_files:
        file_name = os.path.splitext(os.path.basename(file))[0]  # strip .xlsx
        genes = load_gene_ranks(file)

        # Assign a rank within this model (1 = most important)
        for rank, gene in enumerate(genes, start=1):
            if gene not in gene_ranks_sum:
                gene_ranks_sum[gene] = 0
                gene_count[gene] = 0
            gene_ranks_sum[gene] += rank
            gene_count[gene] += 1

    # Compute average rank per gene
    gene_avg_ranks = {
        gene: gene_ranks_sum[gene] / gene_count[gene] for gene in gene_ranks_sum
    }

    # Build result DataFrame
    result_df = pd.DataFrame(
        [
            {"Gene": gene, "Average_Rank": avg, "N_Models": gene_count[gene]}
            for gene, avg in gene_avg_ranks.items()
        ]
    )

    # Sort by average rank (ascending = more consistently important)
    result_df = result_df.sort_values(by="Average_Rank").reset_index(drop=True)

    # Add overall consensus rank column
    result_df.insert(0, "Rank", range(1, len(result_df) + 1))

    # Save to Excel
    result_df.to_excel(OUTPUT_FILE_PATH, index=False, engine="openpyxl")
    print(f"Results saved to {OUTPUT_FILE_PATH}")


if __name__ == "__main__":
    main()