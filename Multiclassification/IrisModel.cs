using Microsoft.ML;
using System;
using System.IO;
using static Microsoft.ML.DataOperationsCatalog;

namespace Multiclassification
{
    class IrisModel
    {
        static readonly string modelDir = "savedModel";
        static readonly string dataPath = Path.Combine(Environment.CurrentDirectory, "data", "flowers.txt");
        static readonly string modelPath = Path.Combine(Environment.CurrentDirectory, modelDir, "IrisClassificationModel.zip");
        static MLContext mlContext = new MLContext();

        public static void TrainModel()
        {
            IDataView dataView = mlContext.Data.LoadFromTextFile<ModelInput>(dataPath, hasHeader: false, separatorChar: ',');
            TrainTestData splitDataView = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);

            string featuresColumnName = "Features";

            var pipeline = mlContext.Transforms
                .Conversion.MapValueToKey(inputColumnName: "FlowerType", outputColumnName: "Label")
                .Append(mlContext.Transforms.Concatenate(
                        featuresColumnName,
                        "SepalLength",
                        "SepalWidth",
                        "PetalLength",
                        "PetalWidth"))
                .Append(mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy())
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            var model = pipeline.Fit(dataView);

            if (!Directory.Exists(modelDir))
            {
                Directory.CreateDirectory(modelDir);
            }

            using (var fileStream = new FileStream(modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
            {
                mlContext.Model.Save(model, dataView.Schema, fileStream);
            }
        }

        public static void Predict(ModelInput feature)
        {
            var model = mlContext.Model.Load(modelPath, out var modelSchema);
            var predictor = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(model);
            var prediction = predictor.Predict(feature);
            Console.WriteLine($"Predicted Class: {prediction.Prediction}");
            Console.WriteLine($"Probability: {string.Join(" ", prediction.Score)}");
        }
    }
}
