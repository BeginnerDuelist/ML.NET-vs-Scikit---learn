using Microsoft.ML.Data;

namespace MLNetBenchmark;

// Adult Census Income - used for classification and clustering.
// Expected column layout after Program.cs pre-cleaning (cleaned-adult.csv):
//   0  age
//   1  workclass
//   2  fnlwgt
//   3  education
//   4  educational-num
//   5  marital-status
//   6  occupation
//   7  relationship
//   8  race
//   9  gender
//   10 capital-gain
//   11 capital-loss
//   12 hours-per-week
//   13 native-country
//   14 income (0 or 1, binary label)
public class AdultRecord
{
    [LoadColumn(0)]
    public float Age;

    [LoadColumn(1)]
    public string Workclass = string.Empty;

    [LoadColumn(2)]
    public float Fnlwgt;

    [LoadColumn(3)]
    public string Education = string.Empty;

    [LoadColumn(4)]
    public float EducationalNum;

    [LoadColumn(5)]
    public string MaritalStatus = string.Empty;

    [LoadColumn(6)]
    public string Occupation = string.Empty;

    [LoadColumn(7)]
    public string Relationship = string.Empty;

    [LoadColumn(8)]
    public string Race = string.Empty;

    [LoadColumn(9)]
    public string Gender = string.Empty;

    [LoadColumn(10)]
    public float CapitalGain;

    [LoadColumn(11)]
    public float CapitalLoss;

    [LoadColumn(12)]
    public float HoursPerWeek;

    [LoadColumn(13)]
    public string NativeCountry = string.Empty;

    [LoadColumn(14), ColumnName("Label")]
    public bool Income;
}

// California Housing - used for regression.
// Expected column layout after Program.cs pre-cleaning (cleaned-housing.csv):
//   0 longitude
//   1 latitude
//   2 housing_median_age
//   3 total_rooms
//   4 total_bedrooms (imputed with median)
//   5 population
//   6 households
//   7 median_income
//   8 ocean_proximity
//   9 median_house_value (Label)
public class HousingRecord
{
    [LoadColumn(0)]
    public float Longitude;

    [LoadColumn(1)]
    public float Latitude;

    [LoadColumn(2)]
    public float HousingMedianAge;

    [LoadColumn(3)]
    public float TotalRooms;

    [LoadColumn(4)]
    public float TotalBedrooms;

    [LoadColumn(5)]
    public float Population;

    [LoadColumn(6)]
    public float Households;

    [LoadColumn(7)]
    public float MedianIncome;

    [LoadColumn(8)]
    public string OceanProximity = string.Empty;

    [LoadColumn(9), ColumnName("Label")]
    public float MedianHouseValue;
}

public class BinaryPrediction
{
    [ColumnName("PredictedLabel")]
    public bool PredictedLabel;

    public float Score;

    public float Probability;
}

public class ClusterPrediction
{
    [ColumnName("PredictedLabel")]
    public uint PredictedClusterId;

    [ColumnName("Score")]
    public float[]? Distances;
}

public class RegressionPrediction
{
    public float Score;
}
