using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.IO;
using System.Linq;
using static Microsoft.ML.DataOperationsCatalog;

namespace Multiclassification
{
    class IrisModel
    {
        // Define path
        static readonly string modelDir = "savedModel";
        static readonly string dataPath = Path.Combine(Environment.CurrentDirectory, "data", "flowers.txt");
        static readonly string modelPath = Path.Combine(Environment.CurrentDirectory, modelDir, "IrisClassificationModel.zip");

        public static void TrainModel(MLContext mlContext)
        {
            IDataView dataView = mlContext.Data.LoadFromTextFile<ModelInput>(dataPath, hasHeader: false, separatorChar: ',');
            TrainTestData splitDataView = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);

            var pipeline = mlContext.Transforms
                .Conversion.MapValueToKey(inputColumnName: nameof(ModelInput.Type), outputColumnName: "Label") // Map the string label into 0,1,2
                .Append(mlContext.Transforms.Concatenate(
                        "Features",
                        nameof(ModelInput.SepalLength),
                        nameof(ModelInput.SepalWidth),
                        nameof(ModelInput.PetalLength),
                        nameof(ModelInput.PetalWidth)))
                .Append(mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy())
                .Append(mlContext.Transforms.Conversion.MapKeyToValue(inputColumnName:"PredictedLabel", outputColumnName:"Type"));

            var model = pipeline.Fit(splitDataView.TrainSet);

            // Model evaluation
            var yPred = model.Transform(splitDataView.TestSet);
            var evaluationMetrics = mlContext.MulticlassClassification.Evaluate(yPred);
            Console.WriteLine("================= Model Evaluation =================");
            Console.WriteLine($"{evaluationMetrics.ConfusionMatrix.GetFormattedConfusionTable()}");
            Console.WriteLine($"Macro Accuracy: {evaluationMetrics.MacroAccuracy}");
            Console.WriteLine($"Micro Accuracy: {evaluationMetrics.MicroAccuracy}");
            Console.WriteLine("====================================================");

            // Save model
            if (!Directory.Exists(modelDir))
            {
                Directory.CreateDirectory(modelDir);
            }

            using (var fileStream = new FileStream(modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
            {
                mlContext.Model.Save(model, dataView.Schema, fileStream);
            }
        }

        public static void Predict(MLContext mlContext, ModelInput feature)
        {
            var model = mlContext.Model.Load(modelPath, out var modelSchema);
            var predictor = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(model);
            var prediction = predictor.Predict(feature);
            Console.WriteLine($"Predicted Class: {prediction.Prediction} - {prediction.Type}");
            Console.WriteLine($"Probability: {string.Join(" ", prediction.Score)}");
        }
    }
}
