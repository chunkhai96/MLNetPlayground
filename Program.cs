using System;
using Microsoft.ML;
using Multiclassification;

namespace MLNetPlayground
{
    class Program
    {
        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext();
            IrisModel.TrainModel(mlContext);
            ModelInput sample = new ModelInput();
            while (true)
            {
                Console.WriteLine("Sepal length: ");
                sample.SepalLength = float.Parse(Console.ReadLine());
                Console.WriteLine("Sepal width: ");
                sample.SepalWidth = float.Parse(Console.ReadLine());
                Console.WriteLine("Petal length: ");
                sample.PetalLength = float.Parse(Console.ReadLine());
                Console.WriteLine("Petal width: ");
                sample.PetalWidth = float.Parse(Console.ReadLine());

                IrisModel.Predict(mlContext, sample);

                Console.WriteLine("Enter q to exit.");
                if (Console.ReadLine().Equals("q")) break;            
            }
        }
    }
}
