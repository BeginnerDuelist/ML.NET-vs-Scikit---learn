# MLNetBenchmark

ML.NET (C# / .NET 8) counterpart of the scikit-learn benchmark. Produces
`results_mlnet.json` with the exact same schema as `results_sklearn.json` so
that `generate_charts.py` can load both files directly.

## Datasets

Two Kaggle datasets are used, each chosen so that both ecosystems produce
naturally comparable metrics:

| File | Dataset | Purpose |
|---|---|---|
| `../data/adult.csv`   | `uciml/adult-census-income`       | classification + clustering |
| `../data/housing.csv` | `camnugent/california-housing-prices` | regression |

Download with the Kaggle CLI (from the repository root):

```bash
kaggle datasets download -d uciml/adult-census-income         -p data --unzip
kaggle datasets download -d camnugent/california-housing-prices -p data --unzip
```

## Prerequisites

- [.NET 8 SDK](https://dotnet.microsoft.com/en-us/download/dotnet/8.0)
- Python 3.10+ with packages from `../requirements.txt`
- The two CSV files above placed in `../data/`

The first build downloads NuGet packages (`Microsoft.ML`, `Microsoft.ML.FastTree`).

## Run order

```bash
# 1. scikit-learn benchmark
python experiments_sklearn.py

# 2. ML.NET benchmark
cd MLNetBenchmark
dotnet run -c Release
cd ..

# 3. charts
python generate_charts.py
```

## What Program.cs does

1. **Pre-clean phase** (pure C# `StreamReader`):
   - `cleaned-adult.csv` - drops rows containing `?`, trims whitespace, converts
     the `income` column to `0` / `1`. Same 15-column layout as the raw file.
   - `cleaned-housing.csv` - imputes missing `total_bedrooms` with the column
     median, reorders the columns so that `ocean_proximity` is column 8 and
     `median_house_value` (the label) is column 9, matching `Models.cs`.
   Both cleaned files are written to `bin/Release/net8.0/cleaned/`.
2. **Classification + clustering pipeline**:
   `LoadFromTextFile<AdultRecord>` -> `OneHotEncoding` (8 categorical columns)
   -> `Concatenate("Features", ...)` (14 inputs total) -> `TrainTestSplit(0.2,
   seed: 42)` -> `NormalizeMinMax("Features")` fit only on the training split
   -> `Cache()`.
3. **Regression pipeline**: `LoadFromTextFile<HousingRecord>` ->
   `OneHotEncoding("OceanProximity")` -> `ReplaceMissingValues("TotalBedrooms",
   Mean)` (safety net; cleaning already imputed the median) ->
   `Concatenate("Features", ...)` -> split -> `NormalizeMinMax` -> `Cache()`.
4. **Benchmarks** (median of 5 `Stopwatch`-timed runs each):
   - `LbfgsLogisticRegression` (maxIterations = 1000)
   - `FastTree` (numberOfTrees = 1, numberOfLeaves = 31)
   - `FastForest` (numberOfTrees = 100, numberOfLeaves = 20)
   - `LinearSvm` (numberOfIterations = 100)
   - `KMeans` (numberOfClusters = 3)
   - `Sdca` regressor (maxIterations = 100)
5. Writes `../results_mlnet.json` (UTF-8, no BOM) and prints a formatted summary.

## Output schema

Keys match `results_sklearn.json` exactly:

```json
{
  "LogisticRegression": { "train_ms": ..., "infer_ms": ..., "Accuracy": ..., "F1": ..., "Precision": ..., "Recall": ..., "ROC_AUC": ... },
  "DecisionTree":       { "train_ms": ..., "infer_ms": ..., "Accuracy": ..., "F1": ..., "Precision": ..., "Recall": ..., "ROC_AUC": ... },
  "RandomForest":       { "train_ms": ..., "infer_ms": ..., "Accuracy": ..., "F1": ..., "Precision": ..., "Recall": ..., "ROC_AUC": ... },
  "SVM_Linear":         { "train_ms": ..., "infer_ms": ..., "Accuracy": ..., "F1": ..., "Precision": ..., "Recall": ..., "ROC_AUC": ... },
  "KMeans":             { "train_ms": ..., "infer_ms": ..., "Silhouette": ..., "Inertia": ... },
  "LinearRegression":   { "train_ms": ..., "infer_ms": ..., "R2": ..., "RMSE": ..., "MAE": ... }
}
```

## Clustering metric mapping

ML.NET does not expose a silhouette score or an sklearn-style inertia. To keep
the JSON schema identical, the program stores:

- `Silhouette`  <- `ClusteringMetrics.DaviesBouldinIndex` (lower is better,
  not numerically equivalent to sklearn's silhouette).
- `Inertia`    <- `ClusteringMetrics.AverageDistance` (mean distance between
  points and their cluster centroid, not sum-of-squares).

The chart generator only plots train/infer time for KMeans so this mapping
only affects the JSON file and the summary table.

## Categorical encoding asymmetry

- scikit-learn uses `OrdinalEncoder` (integer encoding) for categoricals.
- ML.NET uses `OneHotEncoding` (indicator vectors).

Both sides follow the original task spec. This is the main reason linear
classifiers (`LogisticRegression`, `LinearSvm`) can differ slightly between
the two ecosystems - ordinal encoding imposes an artificial ordering that
linear models implicitly weight.

## Benchmarking notes

- The first `Fit` call triggers substantial JIT work. Median of 5 runs
  absorbs this warm-up cost without discarding the first sample.
- Feature preprocessing (`OneHotEncoding` + `Concatenate` + `NormalizeMinMax`)
  is fit once and excluded from per-algorithm timings, matching Python's
  `StandardScaler().fit(X_train)` usage.
- Transformed views are cached with `mlContext.Data.Cache(...)` so repeated
  passes during the 5 runs read from memory instead of re-running the
  pipeline.

## Why two datasets?

The previous iteration used `Give Me Some Credit`, where the positive class
is only ~6.7%. That severe imbalance made weighted F1 misleading and the
default threshold produced `F1 ~= 0` for some ML.NET trainers. Adult Census
Income (~24% positives) gives stable classification metrics in both
ecosystems, and California Housing is the canonical regression benchmark
(expected `R^2 ~ 0.64` for `LinearRegression` / `Sdca`).
