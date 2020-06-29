using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
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

        public static (String, double, ITransformer) TrainModel(MLContext mlContext, IEstimator<ITransformer> modelType, IDataView data)
        {
            Console.WriteLine("Start Training...");
            TrainTestData splitDataView = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);

            var modelPipeline = mlContext.Transforms
                .Conversion.MapValueToKey(inputColumnName: nameof(ModelInput.Type), outputColumnName: "Label") // Map the string label into 0,1,2
                .Append(mlContext.Transforms.Concatenate(
                        "Features",
                        nameof(ModelInput.SepalLength),
                        nameof(ModelInput.SepalWidth),
                        nameof(ModelInput.PetalLength),
                        nameof(ModelInput.PetalWidth)))
                .Append(modelType)
                .Append(mlContext.Transforms.Conversion.MapKeyToValue(inputColumnName:"PredictedLabel", outputColumnName:"Type"));
            
            var trainedModel = modelPipeline.Fit(splitDataView.TrainSet);

            // Model evaluation
            var yPred = trainedModel.Transform(splitDataView.TestSet);
            var evaluationMetrics = mlContext.MulticlassClassification.Evaluate(yPred);
            Console.WriteLine("================= Model Evaluation =================");
            Console.WriteLine($"{evaluationMetrics.ConfusionMatrix.GetFormattedConfusionTable()}");
            Console.WriteLine($"Macro Accuracy: {evaluationMetrics.MacroAccuracy}");
            Console.WriteLine($"Micro Accuracy: {evaluationMetrics.MicroAccuracy}");
            Console.WriteLine("====================================================");
            
            return (modelType.ToString(), evaluationMetrics.MacroAccuracy, trainedModel);
        }

        public static void CrossValidation(MLContext mlContext)
        {
            IDataView dataView = mlContext.Data.LoadFromTextFile<ModelInput>(dataPath, hasHeader: false, separatorChar: ',');
            var modelList = new IEstimator<ITransformer>[]
            {
                mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy(),
                mlContext.MulticlassClassification.Trainers.NaiveBayes(),
                mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy()
            };

            var cvResults = new List<(String, double, ITransformer)>();

            Array.ForEach(modelList, m => cvResults.Add(TrainModel(mlContext, m, dataView)));

            cvResults = cvResults.OrderByDescending(r => r.Item2).ToList();

            Console.WriteLine($"Best Model: {cvResults[0].Item1} -- Accuracy: {cvResults[0].Item2}");

            SaveModel(mlContext, cvResults[0].Item3, dataView.Schema);
        }

        private static void SaveModel(MLContext mlContext, ITransformer model, DataViewSchema dvSchema)
        {
            if (!Directory.Exists(modelDir))
            {
                Directory.CreateDirectory(modelDir);
            }
            // Save model
            using (var fileStream = new FileStream(modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
            {
                mlContext.Model.Save(model, dvSchema, fileStream);
            }

            Console.WriteLine($"Model saved in {modelPath}");
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
