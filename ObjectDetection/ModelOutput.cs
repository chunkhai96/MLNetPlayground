using Microsoft.ML.Data;

namespace MLNetPlayground.ObjectDetection
{
    class ModelOutput
    {
        [ColumnName("grid")]
        public float[] PredictedLabels;
    }
}
