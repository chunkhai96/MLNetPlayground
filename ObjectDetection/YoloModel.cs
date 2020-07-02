using System;
using System.IO;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Linq;
using Microsoft.ML;
using MLNetPlayground.ObjectDetection.YoloParser;
using System.Windows.Forms;
using System.Threading;
using System.Diagnostics;

namespace MLNetPlayground.ObjectDetection
{
    class YoloModel
    {
        static readonly string assetsRelativePath = @"ObjectDetection\assets";
        static readonly string modelFilePath = Path.Combine(Environment.CurrentDirectory, assetsRelativePath, "Model", "tiny_yolov2.onnx");
        static readonly string imagesFolder = Path.Combine(Environment.CurrentDirectory, assetsRelativePath, "images");
        static readonly string outputFolder = Path.Combine(Environment.CurrentDirectory, assetsRelativePath, "images", "output");

        public static void Detect(MLContext mlContext)
        {
            Console.WriteLine("Press any key to proceed to image selection...");
            Console.ReadLine();
            var imagePath = SelectImage();
            if (imagePath != null)
            {
                ModelInput img = new ModelInput();
                img.ImagePath = imagePath;
                img.Label = Path.GetFileName(imagePath);
                IEnumerable<ModelInput> input = new List<ModelInput>() { img };
                Predict(mlContext, input, preview:true);
            }
        }

        public static void DetectBatch(MLContext mlContext, string folderPath = null)
        {
            if ((folderPath == null) || folderPath.Equals(""))
                folderPath = imagesFolder;
            IEnumerable<ModelInput> images = ModelInput.ReadFromFolder(folderPath);
            Predict(mlContext, images);
        }

        private static void Predict(MLContext mlContext, IEnumerable<ModelInput> images, bool preview = false)
        {
            Console.WriteLine($"Load model from {modelFilePath}");
            try
            {
                IDataView imageDataView = mlContext.Data.LoadFromEnumerable(images);
                var modelScorer = new OnnxModelScorer(imagesFolder, modelFilePath, mlContext);

                // Use model to score data
                IEnumerable<float[]> probabilities = modelScorer.Score(imageDataView);
                YoloOutputParser parser = new YoloOutputParser();

                var boundingBoxes =
                    probabilities
                    .Select(probability => parser.ParseOutputs(probability))
                    .Select(boxes => parser.FilterBoundingBoxes(boxes, 5, .5F));

                for (var i = 0; i < images.Count(); i++)
                {
                    string imagePath = images.ElementAt(i).ImagePath;
                    IList<YoloBoundingBox> detectedObjects = boundingBoxes.ElementAt(i);
                    DrawBoundingBox(outputFolder, imagePath, detectedObjects, preview);
                    LogDetectedObjects(imagePath, detectedObjects);
                    Console.WriteLine("========= End of Process..Hit any Key ========");
                    Console.ReadLine();
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
            }
        }

        private static void DrawBoundingBox(string outputImageLocation, string imagePath, 
            IList<YoloBoundingBox> filteredBoundingBoxes, bool preview = false)
        {
            Image image = Image.FromFile(imagePath);

            var originalImageHeight = image.Height;
            var originalImageWidth = image.Width;

            foreach (var box in filteredBoundingBoxes)
            {
                var x = (uint)Math.Max(box.Dimensions.X, 0);
                var y = (uint)Math.Max(box.Dimensions.Y, 0);
                var width = (uint)Math.Min(originalImageWidth - x, box.Dimensions.Width);
                var height = (uint)Math.Min(originalImageHeight - y, box.Dimensions.Height);
                x = (uint)originalImageWidth * x / OnnxModelScorer.ImageNetSettings.imageWidth;
                y = (uint)originalImageHeight * y / OnnxModelScorer.ImageNetSettings.imageHeight;
                width = (uint)originalImageWidth * width / OnnxModelScorer.ImageNetSettings.imageWidth;
                height = (uint)originalImageHeight * height / OnnxModelScorer.ImageNetSettings.imageHeight;
                string text = $"{box.Label} ({(box.Confidence * 100).ToString("0")}%)";
                using (Graphics thumbnailGraphic = Graphics.FromImage(image))
                {
                    thumbnailGraphic.CompositingQuality = CompositingQuality.HighQuality;
                    thumbnailGraphic.SmoothingMode = SmoothingMode.HighQuality;
                    thumbnailGraphic.InterpolationMode = InterpolationMode.HighQualityBicubic;

                    // Define Text Options
                    Font drawFont = new Font("Arial", 12, FontStyle.Bold);
                    SizeF size = thumbnailGraphic.MeasureString(text, drawFont);
                    SolidBrush fontBrush = new SolidBrush(Color.Black);
                    Point atPoint = new Point((int)x, (int)y - (int)size.Height - 1);

                    // Define BoundingBox options
                    Pen pen = new Pen(box.BoxColor, 3.2f);
                    SolidBrush colorBrush = new SolidBrush(box.BoxColor);
                    thumbnailGraphic.FillRectangle(colorBrush, (int)x, (int)(y - size.Height - 1), (int)size.Width, (int)size.Height);
                    thumbnailGraphic.DrawString(text, drawFont, fontBrush, atPoint);

                    // Draw bounding box on image
                    thumbnailGraphic.DrawRectangle(pen, x, y, width, height);
                }
            }

            if (!Directory.Exists(outputImageLocation))
            {
                Directory.CreateDirectory(outputImageLocation);
            }

            var imageName = Path.GetFileName(imagePath);

            image.Save(Path.Combine(outputImageLocation, imageName));

            if (preview)
            {
                var p = new Process();
                p.StartInfo = new ProcessStartInfo(Path.Combine(outputImageLocation, imageName))
                {
                    UseShellExecute = true
                };
                p.Start();
            }
            
        }

        private static void LogDetectedObjects(string imagePath, IList<YoloBoundingBox> boundingBoxes)
        {
            var imageName = Path.GetFileName(imagePath);
            Console.WriteLine($".....The objects in the image {imageName} are detected as below....");

            foreach (var box in boundingBoxes)
            {
                Console.WriteLine($"{box.Label} and its Confidence score: {box.Confidence}");
            }

            Console.WriteLine("");

        }

        private static string SelectImage()
        {
            string imagePath = null;
            OpenFileDialog fileDialog = new OpenFileDialog();
            var t = new Thread((ThreadStart)(() => {
                OpenFileDialog fbd = new OpenFileDialog();
                // image filters  
                fileDialog.Filter = "Image Files(*.jpg; *.jpeg; *.gif; *.bmp)|*.jpg; *.jpeg; *.gif; *.bmp";
                if (fileDialog.ShowDialog() == DialogResult.OK)
                {
                    imagePath = fileDialog.FileName;
                }
            }));

            t.SetApartmentState(ApartmentState.STA);
            t.Start();
            t.Join();

            return imagePath;
        }
    }
}
