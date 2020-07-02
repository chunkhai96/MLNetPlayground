using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.Data;

namespace MLNetPlayground.ObjectDetection
{
    class ModelInput
    {
        [LoadColumn(0)]
        public string ImagePath;

        [LoadColumn(1)]
        public string Label;

        public static IEnumerable<ModelInput> ReadFromFolder(string imageFolder)
        {
            return Directory
                .GetFiles(imageFolder)
                .Where(filePath => Path.GetExtension(filePath) != ".md")
                .Select(filePath => new ModelInput { ImagePath = filePath, Label = Path.GetFileName(filePath) });
        }
    }
}
