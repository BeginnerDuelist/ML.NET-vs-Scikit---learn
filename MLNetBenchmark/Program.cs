using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;

namespace MLNetBenchmark;

public static class Program
{
    private const int RandomSeed = 42;
    private const int NumberOfRuns = 5;
    private const double TestFraction = 0.2;

    private static readonly string[] AdultCategoricalColumns = new[]
    {
        nameof(AdultRecord.Workclass),
        nameof(AdultRecord.Education),
        nameof(AdultRecord.MaritalStatus),
        nameof(AdultRecord.Occupation),
        nameof(AdultRecord.Relationship),
        nameof(AdultRecord.Race),
        nameof(AdultRecord.Gender),
        nameof(AdultRecord.NativeCountry),
    };

    private static readonly string[] AdultNumericColumns = new[]
    {
        nameof(AdultRecord.Age),
        nameof(AdultRecord.Fnlwgt),
        nameof(AdultRecord.EducationalNum),
        nameof(AdultRecord.CapitalGain),
        nameof(AdultRecord.CapitalLoss),
        nameof(AdultRecord.HoursPerWeek),
    };

    private static readonly string[] HousingNumericColumns = new[]
    {
        nameof(HousingRecord.Longitude),
        nameof(HousingRecord.Latitude),
        nameof(HousingRecord.HousingMedianAge),
        nameof(HousingRecord.TotalRooms),
        nameof(HousingRecord.TotalBedrooms),
        nameof(HousingRecord.Population),
        nameof(HousingRecord.Households),
        nameof(HousingRecord.MedianIncome),
    };

    public static void Main(string[] args)
    {
        string rootPath = _ResolveRootPath(args);
        string adultRaw = Path.Combine(rootPath, "data", "adult.csv");
        string housingRaw = Path.Combine(rootPath, "data", "housing.csv");

        if (!File.Exists(adultRaw))
        {
            Console.Error.WriteLine($"Dataset not found: {adultRaw}");
            Console.Error.WriteLine(
                "Download with: kaggle datasets download -d uciml/adult-census-income -p data --unzip");
            Environment.Exit(1);
        }

        if (!File.Exists(housingRaw))
        {
            Console.Error.WriteLine($"Dataset not found: {housingRaw}");
            Console.Error.WriteLine(
                "Download with: kaggle datasets download -d camnugent/california-housing-prices -p data --unzip");
            Environment.Exit(1);
        }

        string cleanedDir = Path.Combine(AppContext.BaseDirectory, "cleaned");
        Directory.CreateDirectory(cleanedDir);
        string adultCleaned = Path.Combine(cleanedDir, "cleaned-adult.csv");
        string housingCleaned = Path.Combine(cleanedDir, "cleaned-housing.csv");

        Console.WriteLine("Pre-cleaning Adult Census Income...");
        int adultRows = _PreCleanAdult(adultRaw, adultCleaned);
        Console.WriteLine($"  cleaned rows: {adultRows}");

        Console.WriteLine("Pre-cleaning California Housing...");
        int housingRows = _PreCleanHousing(housingRaw, housingCleaned);
        Console.WriteLine($"  cleaned rows: {housingRows}");

        MLContext mlContext = new MLContext(seed: RandomSeed);

        Dictionary<string, Dictionary<string, double>> results = new Dictionary<string, Dictionary<string, double>>();

        Console.WriteLine("\nPreparing Adult classification / clustering splits...");
        _PrepareAdultSplits(
            mlContext,
            adultCleaned,
            out IDataView adultTrain,
            out IDataView adultTest);

        Console.WriteLine("\nRunning classifiers...");
        results["LogisticRegression"] = _RunBinaryTrainer(
            mlContext,
            adultTrain,
            adultTest,
            () => mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression(
                labelColumnName: "Label",
                featureColumnName: "Features",
                l1Regularization: 1f,
                l2Regularization: 1f,
                historySize: 20,
                optimizationTolerance: 1e-7f,
                enforceNonNegativity: false),
            calibrated: true);

        results["DecisionTree"] = _RunBinaryTrainer(
            mlContext,
            adultTrain,
            adultTest,
            () => mlContext.BinaryClassification.Trainers.FastTree(
                labelColumnName: "Label",
                featureColumnName: "Features",
                numberOfLeaves: 31,
                numberOfTrees: 1,
                minimumExampleCountPerLeaf: 10,
                learningRate: 0.2),
            calibrated: true);

        results["RandomForest"] = _RunBinaryTrainer(
            mlContext,
            adultTrain,
            adultTest,
            () => mlContext.BinaryClassification.Trainers.FastForest(
                labelColumnName: "Label",
                featureColumnName: "Features",
                numberOfLeaves: 20,
                numberOfTrees: 100,
                minimumExampleCountPerLeaf: 10),
            calibrated: false);

        results["SVM_Linear"] = _RunBinaryTrainer(
            mlContext,
            adultTrain,
            adultTest,
            () => mlContext.BinaryClassification.Trainers.LinearSvm(
                labelColumnName: "Label",
                featureColumnName: "Features",
                numberOfIterations: 100),
            calibrated: false);

        Console.WriteLine("\nRunning clustering...");
        results["KMeans"] = _RunKMeans(mlContext, adultTrain, adultTest);

        Console.WriteLine("\nRunning regression on California Housing...");
        results["LinearRegression"] = _RunRegression(mlContext, housingCleaned);

        string outputPath = Path.Combine(rootPath, "results_mlnet.json");
        _WriteResultsJson(outputPath, results);

        _PrintSummary(results);
        Console.WriteLine($"\nResults written to {outputPath}");
    }

    private static string _ResolveRootPath(string[] args)
    {
        if (args.Length > 0 && Directory.Exists(args[0]))
        {
            return Path.GetFullPath(args[0]);
        }

        string? current = AppContext.BaseDirectory;
        for (int i = 0; i < 8 && current != null; i++)
        {
            if (File.Exists(Path.Combine(current, "data", "adult.csv")))
            {
                return current;
            }

            string? parent = Directory.GetParent(current)?.FullName;
            if (parent != null && File.Exists(Path.Combine(parent, "data", "adult.csv")))
            {
                return parent;
            }

            current = parent;
        }

        return Directory.GetCurrentDirectory();
    }

    private static int _PreCleanAdult(string sourceCsv, string destCsv)
    {
        CultureInfo inv = CultureInfo.InvariantCulture;
        int kept = 0;

        using (StreamReader reader = new StreamReader(sourceCsv))
        using (StreamWriter writer = new StreamWriter(destCsv, append: false, _NoBomUtf8()))
        {
            string? header = reader.ReadLine();
            if (header == null)
            {
                throw new InvalidDataException("Empty adult CSV.");
            }

            // Keep the original 15-column order; only the income column is rewritten as 0/1.
            writer.WriteLine(header);

            string? line;
            while ((line = reader.ReadLine()) != null)
            {
                if (line.Length == 0)
                {
                    continue;
                }

                string[] tokens = line.Split(',');
                if (tokens.Length < 15)
                {
                    continue;
                }

                // The Kaggle uciml/adult-census-income export wraps string
                // values in double quotes (e.g. "?","Private",">50K"). Strip
                // both whitespace and quotes so downstream comparisons work.
                for (int i = 0; i < tokens.Length; i++)
                {
                    tokens[i] = tokens[i].Trim().Trim('"');
                }

                bool hasMissing = false;
                for (int i = 0; i < 15; i++)
                {
                    if (tokens[i] == "?" || tokens[i].Length == 0)
                    {
                        hasMissing = true;
                        break;
                    }
                }

                if (hasMissing)
                {
                    continue;
                }

                string incomeRaw = tokens[14].Replace(".", string.Empty);
                string labelBit = incomeRaw == ">50K" ? "1" : "0";
                tokens[14] = labelBit;

                writer.WriteLine(string.Join(',', tokens));
                kept++;
            }
        }

        return kept;
    }

    private static int _PreCleanHousing(string sourceCsv, string destCsv)
    {
        CultureInfo inv = CultureInfo.InvariantCulture;

        List<string[]> rows = new List<string[]>(capacity: 21_000);
        string[]? originalHeader = null;
        using (StreamReader reader = new StreamReader(sourceCsv))
        {
            originalHeader = reader.ReadLine()?.Split(',');
            if (originalHeader == null)
            {
                throw new InvalidDataException("Empty housing CSV.");
            }

            string? line;
            while ((line = reader.ReadLine()) != null)
            {
                if (line.Length == 0)
                {
                    continue;
                }
                rows.Add(line.Split(','));
            }
        }

        Dictionary<string, int> nameToIndex = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        for (int i = 0; i < originalHeader.Length; i++)
        {
            nameToIndex[originalHeader[i].Trim()] = i;
        }

        int idxLongitude = nameToIndex["longitude"];
        int idxLatitude = nameToIndex["latitude"];
        int idxHousingMedianAge = nameToIndex["housing_median_age"];
        int idxTotalRooms = nameToIndex["total_rooms"];
        int idxTotalBedrooms = nameToIndex["total_bedrooms"];
        int idxPopulation = nameToIndex["population"];
        int idxHouseholds = nameToIndex["households"];
        int idxMedianIncome = nameToIndex["median_income"];
        int idxMedianHouseValue = nameToIndex["median_house_value"];
        int idxOceanProximity = nameToIndex["ocean_proximity"];

        List<double> bedroomObserved = new List<double>(capacity: rows.Count);
        foreach (string[] row in rows)
        {
            if (_TryParseDouble(row[idxTotalBedrooms], out double value))
            {
                bedroomObserved.Add(value);
            }
        }

        double medianBedrooms = _Median(bedroomObserved);

        // Reordered cleaned output:
        //   longitude, latitude, housing_median_age, total_rooms, total_bedrooms,
        //   population, households, median_income, ocean_proximity, median_house_value
        string cleanedHeader = string.Join(
            ',',
            "longitude",
            "latitude",
            "housing_median_age",
            "total_rooms",
            "total_bedrooms",
            "population",
            "households",
            "median_income",
            "ocean_proximity",
            "median_house_value");

        int kept = 0;
        using (StreamWriter writer = new StreamWriter(destCsv, append: false, _NoBomUtf8()))
        {
            writer.WriteLine(cleanedHeader);

            foreach (string[] row in rows)
            {
                double longitude = _ParseOrZero(row[idxLongitude]);
                double latitude = _ParseOrZero(row[idxLatitude]);
                double housingMedianAge = _ParseOrZero(row[idxHousingMedianAge]);
                double totalRooms = _ParseOrZero(row[idxTotalRooms]);

                double totalBedrooms = _TryParseDouble(row[idxTotalBedrooms], out double tb)
                    ? tb
                    : medianBedrooms;

                double population = _ParseOrZero(row[idxPopulation]);
                double households = _ParseOrZero(row[idxHouseholds]);
                double medianIncome = _ParseOrZero(row[idxMedianIncome]);
                double medianHouseValue = _ParseOrZero(row[idxMedianHouseValue]);

                string oceanProximity = row[idxOceanProximity].Trim().Trim('"');

                writer.WriteLine(string.Join(
                    ',',
                    longitude.ToString("G17", inv),
                    latitude.ToString("G17", inv),
                    housingMedianAge.ToString("G17", inv),
                    totalRooms.ToString("G17", inv),
                    totalBedrooms.ToString("G17", inv),
                    population.ToString("G17", inv),
                    households.ToString("G17", inv),
                    medianIncome.ToString("G17", inv),
                    oceanProximity,
                    medianHouseValue.ToString("G17", inv)));
                kept++;
            }
        }

        Console.WriteLine($"  median_total_bedrooms(imputed)={medianBedrooms:F1}");
        return kept;
    }

    private static bool _TryParseDouble(string? token, out double value)
    {
        if (string.IsNullOrWhiteSpace(token))
        {
            value = 0.0;
            return false;
        }

        return double.TryParse(
            token.Trim(),
            NumberStyles.Float,
            CultureInfo.InvariantCulture,
            out value);
    }

    private static double _ParseOrZero(string? token)
    {
        return _TryParseDouble(token, out double v) ? v : 0.0;
    }

    private static double _Median(List<double> data)
    {
        if (data.Count == 0)
        {
            return 0.0;
        }

        double[] sorted = data.ToArray();
        Array.Sort(sorted);
        int mid = sorted.Length / 2;
        if (sorted.Length % 2 == 0)
        {
            return (sorted[mid - 1] + sorted[mid]) / 2.0;
        }
        return sorted[mid];
    }

    private static UTF8Encoding _NoBomUtf8()
    {
        return new UTF8Encoding(encoderShouldEmitUTF8Identifier: false);
    }

    private static void _PrepareAdultSplits(
        MLContext mlContext,
        string cleanedCsv,
        out IDataView trainData,
        out IDataView testData)
    {
        IDataView raw = mlContext.Data.LoadFromTextFile<AdultRecord>(
            cleanedCsv,
            hasHeader: true,
            separatorChar: ',',
            allowQuoting: true);

        InputOutputColumnPair[] oneHotPairs = AdultCategoricalColumns
            .Select(c => new InputOutputColumnPair(c + "Enc", c))
            .ToArray();

        IEstimator<ITransformer> encodingEstimator = mlContext.Transforms.Categorical.OneHotEncoding(
            oneHotPairs,
            outputKind: OneHotEncodingEstimator.OutputKind.Indicator);

        string[] featureColumns = AdultNumericColumns
            .Concat(AdultCategoricalColumns.Select(c => c + "Enc"))
            .ToArray();

        IEstimator<ITransformer> featurePipeline = encodingEstimator
            .Append(mlContext.Transforms.Concatenate("Features", featureColumns));

        ITransformer featureTransformer = featurePipeline.Fit(raw);
        IDataView featurized = featureTransformer.Transform(raw);

        DataOperationsCatalog.TrainTestData split = mlContext.Data.TrainTestSplit(
            featurized,
            testFraction: TestFraction,
            seed: RandomSeed);

        IEstimator<ITransformer> normalizer = mlContext.Transforms.NormalizeMinMax("Features");
        ITransformer normalizerModel = normalizer.Fit(split.TrainSet);

        trainData = mlContext.Data.Cache(normalizerModel.Transform(split.TrainSet));
        testData = mlContext.Data.Cache(normalizerModel.Transform(split.TestSet));
    }

    private static Dictionary<string, double> _RunBinaryTrainer(
        MLContext mlContext,
        IDataView trainData,
        IDataView testData,
        Func<IEstimator<ITransformer>> trainerFactory,
        bool calibrated)
    {
        List<double> trainTimes = new List<double>(NumberOfRuns);
        List<double> inferTimes = new List<double>(NumberOfRuns);
        ITransformer? lastModel = null;

        Stopwatch stopwatch = new Stopwatch();
        for (int i = 0; i < NumberOfRuns; i++)
        {
            IEstimator<ITransformer> trainer = trainerFactory();

            stopwatch.Restart();
            ITransformer model = trainer.Fit(trainData);
            stopwatch.Stop();
            trainTimes.Add(stopwatch.Elapsed.TotalMilliseconds);

            stopwatch.Restart();
            IDataView predictions = model.Transform(testData);
            int count = 0;
            foreach (BinaryPrediction _ in mlContext.Data.CreateEnumerable<BinaryPrediction>(
                predictions,
                reuseRowObject: true,
                ignoreMissingColumns: true))
            {
                count++;
            }
            stopwatch.Stop();
            inferTimes.Add(stopwatch.Elapsed.TotalMilliseconds);

            lastModel = model;
        }

        IDataView finalPredictions = lastModel!.Transform(testData);
        BinaryClassificationMetrics metrics = calibrated
            ? mlContext.BinaryClassification.Evaluate(
                finalPredictions,
                labelColumnName: "Label",
                scoreColumnName: "Score",
                probabilityColumnName: "Probability",
                predictedLabelColumnName: "PredictedLabel")
            : mlContext.BinaryClassification.EvaluateNonCalibrated(
                finalPredictions,
                labelColumnName: "Label",
                scoreColumnName: "Score",
                predictedLabelColumnName: "PredictedLabel");

        return new Dictionary<string, double>
        {
            ["train_ms"] = _MedianMs(trainTimes),
            ["infer_ms"] = _MedianMs(inferTimes),
            ["Accuracy"] = metrics.Accuracy,
            ["F1"] = metrics.F1Score,
            ["Precision"] = metrics.PositivePrecision,
            ["Recall"] = metrics.PositiveRecall,
            ["ROC_AUC"] = metrics.AreaUnderRocCurve,
        };
    }

    private static Dictionary<string, double> _RunKMeans(
        MLContext mlContext,
        IDataView trainData,
        IDataView testData)
    {
        List<double> trainTimes = new List<double>(NumberOfRuns);
        List<double> inferTimes = new List<double>(NumberOfRuns);
        ITransformer? lastModel = null;

        Stopwatch stopwatch = new Stopwatch();
        for (int i = 0; i < NumberOfRuns; i++)
        {
            IEstimator<ITransformer> trainer = mlContext.Clustering.Trainers.KMeans(
                featureColumnName: "Features",
                numberOfClusters: 3);

            stopwatch.Restart();
            ITransformer model = trainer.Fit(trainData);
            stopwatch.Stop();
            trainTimes.Add(stopwatch.Elapsed.TotalMilliseconds);

            stopwatch.Restart();
            IDataView predictions = model.Transform(testData);
            int count = 0;
            foreach (ClusterPrediction _ in mlContext.Data.CreateEnumerable<ClusterPrediction>(
                predictions,
                reuseRowObject: true,
                ignoreMissingColumns: true))
            {
                count++;
            }
            stopwatch.Stop();
            inferTimes.Add(stopwatch.Elapsed.TotalMilliseconds);

            lastModel = model;
        }

        IDataView finalPredictions = lastModel!.Transform(testData);
        ClusteringMetrics metrics = mlContext.Clustering.Evaluate(
            finalPredictions,
            scoreColumnName: "Score",
            featureColumnName: "Features");

        // ML.NET does not expose silhouette score or an sklearn-style inertia.
        // We reuse the same JSON keys by mapping:
        //   Silhouette <- DaviesBouldinIndex   (lower is better, not the same scale)
        //   Inertia    <- AverageDistance      (mean distance to centroid, not sum-of-squares)
        // This mapping is documented in the README.
        return new Dictionary<string, double>
        {
            ["train_ms"] = _MedianMs(trainTimes),
            ["infer_ms"] = _MedianMs(inferTimes),
            ["Silhouette"] = metrics.DaviesBouldinIndex,
            ["Inertia"] = metrics.AverageDistance,
        };
    }

    private static Dictionary<string, double> _RunRegression(
        MLContext mlContext,
        string cleanedCsv)
    {
        IDataView raw = mlContext.Data.LoadFromTextFile<HousingRecord>(
            cleanedCsv,
            hasHeader: true,
            separatorChar: ',',
            allowQuoting: true);

        IEstimator<ITransformer> encoder = mlContext.Transforms.Categorical.OneHotEncoding(
            new[]
            {
                new InputOutputColumnPair(
                    nameof(HousingRecord.OceanProximity) + "Enc",
                    nameof(HousingRecord.OceanProximity)),
            },
            outputKind: OneHotEncodingEstimator.OutputKind.Indicator);

        // The cleaned CSV already imputes missing total_bedrooms with the median.
        // We still include ReplaceMissingValues as a safety net matching the original spec.
        IEstimator<ITransformer> missingFill = mlContext.Transforms.ReplaceMissingValues(
            outputColumnName: nameof(HousingRecord.TotalBedrooms),
            inputColumnName: nameof(HousingRecord.TotalBedrooms),
            replacementMode: MissingValueReplacingEstimator.ReplacementMode.Mean);

        string[] featureColumns = HousingNumericColumns
            .Concat(new[] { nameof(HousingRecord.OceanProximity) + "Enc" })
            .ToArray();

        IEstimator<ITransformer> featurePipeline = encoder
            .Append(missingFill)
            .Append(mlContext.Transforms.Concatenate("Features", featureColumns));

        ITransformer featureTransformer = featurePipeline.Fit(raw);
        IDataView featurized = featureTransformer.Transform(raw);

        DataOperationsCatalog.TrainTestData split = mlContext.Data.TrainTestSplit(
            featurized,
            testFraction: TestFraction,
            seed: RandomSeed);

        IEstimator<ITransformer> normalizer = mlContext.Transforms.NormalizeMinMax("Features");
        ITransformer normalizerModel = normalizer.Fit(split.TrainSet);
        IDataView trainData = mlContext.Data.Cache(normalizerModel.Transform(split.TrainSet));
        IDataView testData = mlContext.Data.Cache(normalizerModel.Transform(split.TestSet));

        List<double> trainTimes = new List<double>(NumberOfRuns);
        List<double> inferTimes = new List<double>(NumberOfRuns);
        ITransformer? lastModel = null;

        Stopwatch stopwatch = new Stopwatch();
        for (int i = 0; i < NumberOfRuns; i++)
        {
            IEstimator<ITransformer> trainer = mlContext.Regression.Trainers.Sdca(
                labelColumnName: "Label",
                featureColumnName: "Features",
                maximumNumberOfIterations: 100);

            stopwatch.Restart();
            ITransformer model = trainer.Fit(trainData);
            stopwatch.Stop();
            trainTimes.Add(stopwatch.Elapsed.TotalMilliseconds);

            stopwatch.Restart();
            IDataView predictions = model.Transform(testData);
            int count = 0;
            foreach (RegressionPrediction _ in mlContext.Data.CreateEnumerable<RegressionPrediction>(
                predictions,
                reuseRowObject: true,
                ignoreMissingColumns: true))
            {
                count++;
            }
            stopwatch.Stop();
            inferTimes.Add(stopwatch.Elapsed.TotalMilliseconds);

            lastModel = model;
        }

        IDataView finalPredictions = lastModel!.Transform(testData);
        RegressionMetrics metrics = mlContext.Regression.Evaluate(
            finalPredictions,
            labelColumnName: "Label",
            scoreColumnName: "Score");

        return new Dictionary<string, double>
        {
            ["train_ms"] = _MedianMs(trainTimes),
            ["infer_ms"] = _MedianMs(inferTimes),
            ["R2"] = metrics.RSquared,
            ["RMSE"] = metrics.RootMeanSquaredError,
            ["MAE"] = metrics.MeanAbsoluteError,
        };
    }

    private static double _MedianMs(List<double> samples)
    {
        if (samples.Count == 0)
        {
            return 0.0;
        }

        double[] sorted = samples.ToArray();
        Array.Sort(sorted);
        int mid = sorted.Length / 2;
        if (sorted.Length % 2 == 0)
        {
            return (sorted[mid - 1] + sorted[mid]) / 2.0;
        }
        return sorted[mid];
    }

    private static void _WriteResultsJson(
        string outputPath,
        Dictionary<string, Dictionary<string, double>> results)
    {
        JsonSerializerOptions options = new JsonSerializerOptions
        {
            WriteIndented = true,
        };

        string json = JsonSerializer.Serialize(results, options);
        File.WriteAllText(outputPath, json, _NoBomUtf8());
    }

    private static void _PrintSummary(Dictionary<string, Dictionary<string, double>> results)
    {
        Console.WriteLine();
        Console.WriteLine(new string('=', 92));
        Console.WriteLine(
            $"{"Algorithm",-22}{"Train (ms)",14}{"Infer (ms)",14}"
            + $"{"Metric 1",16}{"Metric 2",14}{"Metric 3",14}");
        Console.WriteLine(new string('-', 92));

        Dictionary<string, (string m1, string m2, string? m3)> metricLabels =
            new Dictionary<string, (string, string, string?)>
            {
                ["LogisticRegression"] = ("Accuracy", "F1", "ROC_AUC"),
                ["DecisionTree"] = ("Accuracy", "F1", "ROC_AUC"),
                ["RandomForest"] = ("Accuracy", "F1", "ROC_AUC"),
                ["SVM_Linear"] = ("Accuracy", "F1", "ROC_AUC"),
                ["KMeans"] = ("Silhouette", "Inertia", null),
                ["LinearRegression"] = ("R2", "RMSE", "MAE"),
            };

        foreach (KeyValuePair<string, Dictionary<string, double>> entry in results)
        {
            (string m1Key, string m2Key, string? m3Key) = metricLabels[entry.Key];
            Dictionary<string, double> m = entry.Value;
            string m1 = $"{m1Key}={m[m1Key]:F4}";
            string m2 = $"{m2Key}={m[m2Key]:F4}";
            string m3 = m3Key != null ? $"{m3Key}={m[m3Key]:F4}" : "";

            Console.WriteLine(
                $"{entry.Key,-22}{m["train_ms"],14:F2}{m["infer_ms"],14:F2}"
                + $"{m1,16}{m2,14}{m3,14}");
        }

        Console.WriteLine(new string('=', 92));
    }
}
