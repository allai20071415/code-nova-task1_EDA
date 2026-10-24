"""Task 1: Exploratory Data Analysis using scikit-learn's public Wine dataset."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid")

def main():
    bunch = load_wine(as_frame=True)
    df = bunch.data.copy()
    df["target"] = bunch.target
    df["target_name"] = df["target"].map(dict(enumerate(bunch.target_names)))

    # Dataset profiling
    profile = {
        "rows": int(df.shape[0]),
        "columns_including_target_and_name": int(df.shape[1]),
        "feature_count": int(bunch.data.shape[1]),
        "missing_values_by_column": {k: int(v) for k, v in df.isna().sum().items()},
        "duplicate_rows": int(df.duplicated().sum()),
        "class_counts": {str(k): int(v) for k, v in df["target_name"].value_counts().items()},
        "numeric_summary": df.select_dtypes(include=np.number).describe().round(3).to_dict()
    }
    (OUT / "data_profile.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")
    df.to_csv(OUT / "wine_dataset_inspected.csv", index=False)

    # Cleaning policy: report missing/duplicates, only remove duplicates if found.
    cleaned = df.drop_duplicates().copy()
    # Dataset should have no missing values; if any appear, median-impute numeric columns.
    numeric_cols = cleaned.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        if cleaned[col].isna().any():
            cleaned[col] = cleaned[col].fillna(cleaned[col].median())

    # Missing value chart
    plt.figure(figsize=(12, 5))
    missing = df.isna().sum().sort_values(ascending=False)
    plt.bar(missing.index, missing.values)
    plt.title("Missing Values by Column")
    plt.xlabel("Column")
    plt.ylabel("Missing value count")
    plt.xticks(rotation=70, ha="right")
    plt.tight_layout()
    plt.savefig(OUT / "missing_values.png", dpi=180)
    plt.close()

    # Feature distributions
    cols = ["alcohol", "malic_acid", "ash", "flavanoids", "color_intensity", "proline"]
    cleaned[cols].hist(figsize=(12, 8), bins=18, edgecolor="black")
    plt.suptitle("Distributions of Selected Wine Features", y=1.02)
    plt.tight_layout()
    plt.savefig(OUT / "feature_distributions.png", dpi=180, bbox_inches="tight")
    plt.close()

    # Correlation heatmap (numeric features only)
    plt.figure(figsize=(13, 10))
    corr = cleaned.select_dtypes(include=np.number).corr()
    sns.heatmap(corr, cmap="coolwarm", center=0, linewidths=0.25)
    plt.title("Correlation Heatmap — Wine Dataset")
    plt.tight_layout()
    plt.savefig(OUT / "correlation_heatmap.png", dpi=180)
    plt.close()

    # Boxplots for possible outliers
    box_cols = ["alcohol", "malic_acid", "magnesium", "flavanoids", "color_intensity"]
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=cleaned[box_cols], orient="h")
    plt.title("Potential Outliers in Selected Features (IQR Visual Check)")
    plt.xlabel("Feature value")
    plt.tight_layout()
    plt.savefig(OUT / "outlier_boxplots.png", dpi=180)
    plt.close()

    # Pairwise scatter plot
    pair = sns.pairplot(cleaned, vars=["alcohol", "malic_acid", "flavanoids", "color_intensity"],
                        hue="target_name", corner=True, diag_kind="hist", plot_kws={"alpha": 0.7, "s": 24})
    pair.fig.suptitle("Pairwise Relationships by Wine Class", y=1.02)
    pair.savefig(OUT / "pairwise_scatter.png", dpi=160)
    plt.close("all")

    # IQR counts to inform, not automatically delete, outliers.
    outlier_counts = {}
    for col in bunch.data.columns:
        q1, q3 = cleaned[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        mask = (cleaned[col] < q1 - 1.5 * iqr) | (cleaned[col] > q3 + 1.5 * iqr)
        outlier_counts[col] = int(mask.sum())
    (OUT / "iqr_outlier_counts.json").write_text(json.dumps(outlier_counts, indent=2), encoding="utf-8")

    print("Task 1 EDA completed.")
    print(f"Dataset shape before cleaning: {df.shape}")
    print(f"Duplicate rows: {profile['duplicate_rows']}")
    print(f"Missing values total: {int(df.isna().sum().sum())}")
    print("Cleaning decision: duplicates removed if present; numeric missing values median-imputed if present.")
    print(f"Outputs saved to: {OUT}")
    print("Review the charts and write your own empirical takeaways in the notebook/README.")

if __name__ == "__main__":
    main()
