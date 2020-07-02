using System;
using Microsoft.ML;
//using Multiclassification;
using MLNetPlayground.ObjectDetection;

namespace MLNetPlayground
{
    class Program
    {
        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext();
            YoloModel.Detect(mlContext);   
            //IrisModel.CrossValidation(mlContext);
            //ModelInput sample = new ModelInput();
            //while (true)
            //{
            //    Console.WriteLine("sepal length: ");
            //    sample.SepalLength = float.Parse(Console.ReadLine());
            //    Console.WriteLine("sepal width: ");
            //    sample.SepalWidth = float.Parse(Console.ReadLine());
            //    Console.WriteLine("petal length: ");
            //    sample.PetalLength = float.Parse(Console.ReadLine());
            //    Console.WriteLine("petal width: ");
            //    sample.PetalWidth = float.Parse(Console.ReadLine());

            //    IrisModel.Predict(mlContext, sample);

            //    Console.WriteLine("enter q to exit.");
            //    if (Console.ReadLine().Equals("q")) break;            
            //}
        }
    }
}
