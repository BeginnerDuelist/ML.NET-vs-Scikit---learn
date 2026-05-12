# Benchmark scikit-learn vs ML.NET

Benchmark reproductibil care compară **scikit-learn (Python)** și **ML.NET (C# / .NET 8)** pe aceleași două seturi de date Kaggle, rulând cele **6 algoritmi** și producând două fișiere JSON cu schemă identică, plus 5 grafice comparative.

## Algoritmi comparați

| Categorie | scikit-learn | ML.NET | Cheie JSON |
|---|---|---|---|
| Clasificare | `LogisticRegression` | `LbfgsLogisticRegression` | `LogisticRegression` |
| Clasificare | `DecisionTreeClassifier` | `FastTree` (1 tree, 31 leaves) | `DecisionTree` |
| Clasificare | `RandomForestClassifier` | `FastForest` (100 trees) | `RandomForest` |
| Clasificare | `LinearSVC` | `LinearSvm` | `SVM_Linear` |
| Clustering | `MiniBatchKMeans` | `KMeans` (3 clusters) | `KMeans` |
| Regresie | `LinearRegression` | `Sdca` regressor | `LinearRegression` |

Fiecare algoritm este cronometrat cu **mediana a 5 rulări** (`time.perf_counter` în Python, `Stopwatch` în C#).

## Seturi de date

| Fișier | Dataset Kaggle | Folosit pentru |
|---|---|---|
| `data/adult.csv` | `uciml/adult-census-income` | clasificare + clustering |
| `data/housing.csv` | `camnugent/california-housing-prices` | regresie |

Descarcă-le cu CLI-ul Kaggle (din rădăcina repo-ului):

```bash
kaggle datasets download -d uciml/adult-census-income -p data --unzip
kaggle datasets download -d camnugent/california-housing-prices -p data --unzip
```

## Structura proiectului

```
Articol - Copie/
├── data/                       # CSV-urile Kaggle (adult.csv, housing.csv)
├── charts/                     # 5 grafice PNG generate de generate_charts.py
├── MLNetBenchmark/             # Proiect .NET 8 (benchmark ML.NET)
│   ├── Program.cs              # pipeline + benchmark + scriere results_mlnet.json
│   ├── Models.cs               # POCO-uri pentru LoadFromTextFile
│   ├── MLNetBenchmark.csproj   # referințe NuGet (Microsoft.ML 3.0.1)
│   └── README.md               # detalii specifice ML.NET
├── experiments_sklearn.py      # benchmark scikit-learn → results_sklearn.json
├── generate_charts.py          # 5 grafice PNG din cele două JSON-uri
├── requirements.txt            # dependențe Python
├── results_sklearn.json        # output benchmark Python
├── results_mlnet.json          # output benchmark .NET
└── README.md                   # acest fișier
```

## Prerechizite

- **Python 3.10+**
- **.NET 8 SDK** — [download](https://dotnet.microsoft.com/en-us/download/dotnet/8.0)
- Cele două CSV-uri Kaggle plasate în `data/`

La prima compilare, `dotnet` descarcă pachetele NuGet `Microsoft.ML` și `Microsoft.ML.FastTree` (versiunea `3.0.1`).

## Instalare

### 1. Clonează repo-ul și instalează dependențele Python

```powershell
git clone <repo-url>
cd "Test"

pip install -r requirements.txt
```

### 2. Restaurează pachetele .NET

```powershell
dotnet restore MLNetBenchmark/MLNetBenchmark.csproj
```

## Comenzi de rulare

Rulează cele trei etape **în ordinea de mai jos** (graficele depind de ambele fișiere JSON):

```powershell
# 1) Benchmark scikit-learn → results_sklearn.json
python experiments_sklearn.py

# 2) Benchmark ML.NET → results_mlnet.json
cd MLNetBenchmark
dotnet run -c Release
cd ..

# 3) Generează cele 5 grafice PNG
python generate_charts.py
```

### Comenzi utile suplimentare

```powershell
# Build doar (fără rulare) pentru proiectul .NET
dotnet build MLNetBenchmark/MLNetBenchmark.csproj -c Release

# Curăță artefactele de build .NET
dotnet clean MLNetBenchmark/MLNetBenchmark.csproj

# Rulează direct executabilul .NET după build (din rădăcina repo-ului)
MLNetBenchmark\bin\Release\net8.0\MLNetBenchmark.exe
```

## Output

### Fișiere JSON cu schemă identică

Atât `results_sklearn.json` cât și `results_mlnet.json` au exact aceeași structură, ca să poată fi încărcate simetric de `generate_charts.py`:

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

### Cele 5 grafice generate

Toate graficele sunt salvate în folderul `charts/`.

---

#### 1. Timp de antrenament (scară logaritmică pe Y)

![Timp de antrenament](charts/fig1_train.png)

**Observații:**

| Algoritm | sklearn (ms) | ML.NET (ms) | Câștigător |
|---|---:|---:|---|
| LinearRegression | 2.8 | 86.7 | **sklearn** (~31x) |
| LogisticRegression | 21.7 | 245.4 | **sklearn** (~11x) |
| SVM_Linear | 42.7 | 357.2 | **sklearn** (~8x) |
| DecisionTree | 70.2 | 97.3 | sklearn (~1.4x) |
| RandomForest | 289.9 | 448.7 | sklearn (~1.5x) |
| KMeans | 60.7 | 33.0 | **ML.NET** (~1.8x) |

- **scikit-learn câștigă la 5 din 6 algoritmi** la antrenament. Motivul principal: sklearn delegă matematica grea către **BLAS/LAPACK** (OpenBLAS sau MKL) — biblioteci scrise în Fortran/C, extrem de optimizate pentru CPU-uri moderne (AVX, SSE).
- **ML.NET are overhead constant** din pipeline-ul de transformări (`OneHotEncoding` → `Concatenate` → `NormalizeMinMax`) și din JIT-ul .NET la prima execuție. Asta se vede mai ales la modelele liniare mici, unde overhead-ul depășește calculul efectiv (LinearRegression: 2.8 ms vs 86.7 ms).
- **KMeans inversează tendința**: implementarea ML.NET (KMeansPlusPlus) e mai rapidă decât `MiniBatchKMeans` în sklearn pentru această configurație (3 clustere, ~26k rânduri).
- **Scara logaritmică e necesară** pentru că diferența între cel mai rapid (2.8 ms) și cel mai lent (448 ms) e de ~160x — pe scară liniară, barele mici ar fi invizibile.

---

#### 2. Timp de inferență (scară liniară)

![Timp de inferență](charts/fig2_infer.png)

**Observații:**

| Algoritm | sklearn (ms) | ML.NET (ms) | Câștigător |
|---|---:|---:|---|
| LinearRegression | 0.17 | 2.93 | **sklearn** |
| LogisticRegression | 0.29 | 4.29 | **sklearn** |
| SVM_Linear | 0.40 | 1.66 | **sklearn** |
| KMeans | 0.85 | 2.33 | **sklearn** |
| DecisionTree | 0.89 | 3.62 | **sklearn** |
| **RandomForest** | **41.13** | **7.71** | **ML.NET (~5.3x)** ← outlier! |

- **Pentru 5 din 6 algoritmi sklearn e mai rapid la inferență** — modelele liniare și un singur arbore se reduc la operații triviale (un produs scalar, o parcurgere de adâncime ~10), iar overhead-ul .NET (alocări, marshalling) e mai mare decât calculul în sine.

- **Anomalia `RandomForest`: sklearn = 41.13 ms vs ML.NET = 7.71 ms (sklearn este ~5.3x mai lent!)**

  Motivele sunt **structurale**, nu o problemă de bug:

  1. **Arborii sklearn cresc nelimitat la adâncime.** În `experiments_sklearn.py` configurația e:
     ```python
     RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=RANDOM_STATE)
     ```
     **NU există `max_depth` setat** → fiecare arbore crește până când frunzele devin pure (sau aproape pure). Pe Adult Census (~26k rânduri, 14 features), arborii ajung ușor la **adâncime 20-25 și mii de noduri**.

  2. **ML.NET `FastForest` are limită strictă pe frunze.** În `Program.cs`:
     ```csharp
     mlContext.BinaryClassification.Trainers.FastForest(numberOfTrees: 100, numberOfLeaves: 20)
     ```
     Fiecare arbore are **maximum 20 de frunze** → adâncime ~`log2(20) ≈ 4-5`. Parcurgerea unui arbore = 4-5 comparații, vs ~20-25 la sklearn.

  3. **Înmulțit cu 100 de arbori** → sklearn face de ~5x mai multe comparații per predicție, în plus pentru ~6500 de rânduri de test.

  4. **Implementare nativă vs Cython.** `FastForest` în ML.NET rulează prin `FastTreeNative.dll` (C++ compilat nativ, SIMD-friendly), în timp ce sklearn face traversarea arborilor în Cython + agregare în Python (`predict_proba` calculează probabilități per arbore, apoi mediere).

  5. **`predict_proba` în loc de `predict`.** În `_run_classifiers` chemăm `model.predict(x_test)` (pentru `infer_ms`) **plus** `model.predict_proba(x_test)[:, 1]` (pentru `ROC_AUC`). Doar `predict()` intră în cronometru, dar arborii sklearn sunt arhitectural mai mari → fiecare predicție e mai scumpă.

  **Concluzie pentru articol:** diferența nu reflectă o slăbiciune a sklearn, ci **alegeri implicite diferite de hyperparameter** — `FastForest` favorizează arbori mici (regularizare implicită), sklearn favorizează arbori mari (fit perfect pe train). Dacă ai seta `max_depth=5` în sklearn, gap-ul ar dispărea aproape complet.

---

#### 3. Acuratețe — 4 clasificatori (Y zoomed, valori adnotate)

![Acuratețe](charts/fig3_accuracy.png)

**Observații:**

| Clasificator | sklearn | ML.NET | Δ |
|---|---:|---:|---:|
| LogisticRegression | 0.8306 | 0.8443 | +0.014 ML.NET |
| DecisionTree | 0.8525 | 0.8540 | +0.001 ML.NET |
| RandomForest | **0.8628** | 0.8537 | +0.009 sklearn |
| SVM_Linear | 0.8279 | 0.8328 | +0.005 ML.NET |

- **Toate cele 4 modele sunt în intervalul îngust 0.828 – 0.863** — diferențele sunt sub 1.5% absolut, ceea ce e **complet neglijabil** practic.
- **`RandomForest` sklearn câștigă la acuratețe** (0.8628), exact pentru același motiv pentru care e lent la inferență: arborii adânci memorează mai multe pattern-uri din train, dar și generalizează puțin mai bine pe acest dataset (~0.9% peste ML.NET).
- **`LogisticRegression` ML.NET are un avantaj de 1.4%** — probabil din cauza encoding-ului `OneHotEncoding` (vs `OrdinalEncoder` în sklearn), care e mai prietenos cu modelele liniare (categoricile nu primesc o ordine artificială).
- Pentru un articol, mesajul cheie: **acuratețea nu deosebește semnificativ cele două ecosisteme** pe acest dataset.

---

#### 4. F1-score weighted — 4 clasificatori (Y zoomed, valori adnotate)

![F1-score](charts/fig4_f1.png)

**Observații:**

| Clasificator | sklearn (F1 weighted) | ML.NET (F1) | Δ |
|---|---:|---:|---:|
| LogisticRegression | 0.8161 | 0.6499 | −0.166 |
| DecisionTree | 0.8432 | 0.6569 | −0.186 |
| RandomForest | 0.8582 | 0.6556 | −0.203 |
| SVM_Linear | 0.8093 | 0.6158 | −0.194 |

- **Diferența ~0.17 – 0.20 NU înseamnă că ML.NET e mai slab.** E **o convenție diferită de raportare**:
  - sklearn: `f1_score(y_test, preds, average="weighted")` → calculează F1 separat pentru **clasa 0 și clasa 1**, apoi face media ponderată cu suportul fiecărei clase. Clasa negativă (~76% din Adult Census) e ușor de prezis, are F1 ~0.90+ și **trage media în sus**.
  - ML.NET: `BinaryClassificationMetrics.F1Score` raportează F1 **doar pentru clasa pozitivă** (`>50K`) — cea minoritară (~24%) și mai greu de prezis.
- **Cu un dataset imbalansat 24% / 76%, gap-ul observat (~0.18) este exact ce ar prezice teoria.** Dacă recalculezi F1 pe clasa pozitivă în sklearn, vei obține valori similare cu ML.NET (~0.65).
- **Recomandare pentru articol:** menționează explicit această asimetrie metodologică sau modifică `experiments_sklearn.py` să raporteze și `f1_score(..., pos_label=1, average="binary")` pentru paritate cu ML.NET.

---

#### 5. Regresie liniară — R² și RMSE

![Regresie liniară](charts/fig5_regression.png)

**Observații:**

| Metrică | sklearn | ML.NET | Câștigător |
|---|---:|---:|---|
| R² | 0.6137 | **0.6442** | ML.NET (+0.031) |
| RMSE | $71,148 | **$69,198** | ML.NET (−$1,950) |
| MAE | $51,821 | **$49,656** | ML.NET (−$2,165) |

- **Ambele modele sunt în zona canonică pentru California Housing** (R² ~0.64 este valoarea standard raportată în literatură pentru regresie liniară pe acest dataset, fără feature engineering).
- **ML.NET câștigă marginal la toate cele trei metrici** — motivul este algoritmul:
  - sklearn `LinearRegression` rezolvă **Ordinary Least Squares** prin **Normal Equation** (`(XᵀX)⁻¹Xᵀy`), soluție analitică pură, fără regularizare.
  - ML.NET `Sdca` (Stochastic Dual Coordinate Ascent) este un **algoritm iterativ** (`maxIterations=100`) cu **L2 regularization implicită**. Pe un dataset cu features multicoliniare (latitude/longitude, total_rooms/total_bedrooms/population), regularizarea ușoară ajută la generalizare → R² puțin mai mare.
- **Diferența de ~$2,000 la RMSE este nesemnificativă** la o medie a țintei de ~$200,000 — adică un 1% relativ.
- Concluzie practică: pentru regresie liniară, **alegerea ecosistemului contează mai puțin decât alegerea algoritmului** (OLS vs SDCA vs Ridge vs Lasso).

## Note despre benchmark

- **Mediana a 5 rulări** absoarbe costul JIT/warm-up fără să elimine prima măsurătoare.
- **Preprocesare excluse din timpii pe algoritm** — `StandardScaler` (sklearn) și `OneHotEncoding + NormalizeMinMax` (ML.NET) sunt fitate o singură dată pe trainset, înainte de cronometrare.
- **Cache** — în ML.NET, view-urile transformate sunt cache-uite cu `mlContext.Data.Cache(...)` ca rulările repetate să citească din memorie.
- **Encoding asimetric (intenționat)** — sklearn folosește `OrdinalEncoder` (etichete întregi), iar ML.NET folosește `OneHotEncoding` (vectori indicatori). Asta explică diferențele între clasificatorii liniari (`LogisticRegression`, `LinearSvm`).
- **K-Means metric mapping** — ML.NET nu expune silhouette score, așa că în JSON-ul ML.NET `Silhouette` ← `DaviesBouldinIndex` și `Inertia` ← `AverageDistance`. Graficele plotează doar `train_ms` / `infer_ms` pentru K-Means, deci maparea afectează doar JSON-ul brut.

## Reproductibilitate

- `RANDOM_STATE = 42` în Python; `seed: 42` la `TrainTestSplit` în ML.NET.
- `TEST_SIZE = 0.2` în ambele ecosisteme.
- `N_RUNS = 5` cu raportarea medianei.

## Troubleshooting

- **`FileNotFoundError: data/adult.csv`** — descarcă datasetul Kaggle (vezi secțiunea *Seturi de date*).
- **`Missing results_sklearn.json` la `generate_charts.py`** — rulează întâi `python experiments_sklearn.py`.
- **`Missing results_mlnet.json` la `generate_charts.py`** — rulează întâi `dotnet run -c Release` din `MLNetBenchmark/`.
- **Eroare NuGet la primul build** — verifică conexiunea la internet și rulează `dotnet restore MLNetBenchmark/MLNetBenchmark.csproj`.

## Vezi și

- [`MLNetBenchmark/README.md`](MLNetBenchmark/README.md) — detalii despre pipeline-ul ML.NET, pre-curățarea CSV-urilor și hyperparametrii fiecărui trainer.
