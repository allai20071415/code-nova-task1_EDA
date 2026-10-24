# code-nova-task1_EDA# 
Task 1 — Exploratory Data Analysis (EDA)

## Objective
Analyze a public dataset, assess data quality, visualize feature distributions and relationships, and document empirical observations.

## Dataset
The Wine recognition dataset bundled with scikit-learn. It has 178 samples, 13 numeric chemical measurements, and 3 wine classes. The script prints the actual shape and quality checks when run.

## Run
From this folder:
```powershell
python run_eda.py
```
For the required notebook deliverable, open `EDA_Analysis.ipynb` in VS Code, select the project Python kernel, and choose **Run All**. Save the notebook after execution.

## Generated outputs
- `outputs/data_profile.json` — dimensions, missing values, duplicates, class counts, numeric summary.
- `outputs/wine_dataset_inspected.csv` — inspected dataset.
- `outputs/missing_values.png`
- `outputs/feature_distributions.png`
- `outputs/correlation_heatmap.png`
- `outputs/outlier_boxplots.png`
- `outputs/pairwise_scatter.png`
- `outputs/iqr_outlier_counts.json`

## Cleaning rationale
The script checks missing values and duplicates first. Duplicate rows are removed only if found. Numeric missing values, if present, are median-imputed because the median is less affected by extreme values than the mean. IQR outliers are counted and visualized but are not automatically deleted: an extreme chemical measurement may be a valid observation and should not be removed without domain evidence.

## Interpretation prompts
After running, add observations based on your own figures:
1. Which features show the widest spread?
2. Which feature pairs appear strongly correlated?
3. Do the classes appear separated for any selected feature pair?
4. Which features show IQR outliers, and why should they be investigated rather than automatically removed?
5. Did the data-quality checks find missing values or duplicates?

Do not invent findings: refer to the saved figures and profile JSON.
