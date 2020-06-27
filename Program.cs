using System;
using Multiclassification;

namespace MLNetPlayground
{
    class Program
    {
        static void Main(string[] args)
        {
            IrisModel.TrainModel();
            ModelInput sample = new ModelInput
            {
                SepalLength = 5.9f,
                SepalWidth = 3.0f,
                PetalLength = 5.4f,
                PetalWidth = 1.8f
            };
            IrisModel.Predict(sample);
            Console.ReadLine();
        }
    }
}
