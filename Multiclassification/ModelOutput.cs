using Microsoft.ML.Data;
using System;

namespace Multiclassification
{
    class ModelOutput
    {

        [ColumnName("PredictedLabel")]
        public String Prediction;

        [ColumnName("Prob")]
        public float Probability;

        [ColumnName("Score")]
        public float[] Score;

    }
}
