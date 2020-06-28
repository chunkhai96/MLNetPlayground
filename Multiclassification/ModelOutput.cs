using Microsoft.ML.Data;
using System;

namespace Multiclassification
{
    class ModelOutput
    {

        [ColumnName("PredictedLabel")]
        public uint Prediction;

        [ColumnName("Type")]
        public String Type;

        [ColumnName("Score")]
        public float[] Score;

    }
}
