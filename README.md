# CBAM-VGG16 for Pneumonia Detection 🩺💡

## Description 📝
This repository implements **CBAM-VGG16** for Pneumonia Detection using deep learning techniques. This model integrates the **Convolutional Block Attention Module (CBAM)** with the **VGG16 backbone** to enhance feature extraction, improving pneumonia classification accuracy from Chest X-ray images.  

By leveraging **spatial and channel-wise attention mechanisms**, the model effectively focuses on crucial features, leading to more precise and reliable detection results. 🔬📊

## Features ✨
- 🤖 **Deep Learning-Based Pneumonia Detection** using an enhanced VGG16 architecture.
- 🔍 **CBAM Integration** to improve feature extraction and classification accuracy.
- 🏥 **Trained on Chest X-ray Images** for real-world applicability.
- ⚡ **Optimized Performance** through attention-based feature refinement.

## About **Google Colab Notebook** 📚
The **Google Colab Notebook** presents an original research study on pneumonia detection using deep learning:
- 📌 **Comparative Study:** Compares AlexNet, VGG16, VGG19, ResNet50, and EfficientNet B0 for pneumonia detection on X-ray images.
- 🎯 **CBAM-Enhanced VGG16:** Integrates CBAM for superior feature extraction, accuracy, and F1-score.
- 🚀 **Optimized Training:** Employs hyperparameter tuning and advanced callbacks.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ayusharma24/CBAM-VGG16-Pneumonia-Detection/blob/main/CBAM_VGG16_Pneumonia.ipynb)

## 🛠 Installation & Setup
```bash
# Clone the repository
git clone https://github.com/Ayusharma24/CBAM-VGG16-Pneumonia-Detection.git
cd CBAM-VGG16-Pneumonia-Detection

# Install required dependencies
pip install -r requirements.txt
```

## Usage 🚀
### Steps to Use the Pre-Trained Model in Google Colab:
Note: The **CBAM-VGG16 model.h5** file contains trained weights and is too large to be uploaded directly. To generate this file, simply run the entire *CBAM section* in the **CBAM_VGG16_Pneumonia.ipynb** notebook. This will train the model and save the weights locally.

1. **Load the trained model 🧠:**
   ```python
   from tensorflow.keras.models import load_model
   model = load_model('CBAM-VGG16 model.h5')
   ```

2. **Upload and Preprocess New X-ray Images 🖼️:**
   - 📤 Upload image(s):
     ```python
     from google.colab import files
     uploaded = files.upload()
     ```
   - 🛠 Preprocess images:
     ```python
     import cv2
     import numpy as np
     from tensorflow.keras.preprocessing.image import img_to_array

     def preprocess_image(image_path):
         img = cv2.imread(image_path)
         img = cv2.resize(img, (224, 224))
         img = img_to_array(img) / 255.0
         img = np.expand_dims(img, axis=0)
         return img
     ```

3. **Run Predictions 🔮:**
   ```python
   image_path = 'sample_xray.jpg'  # Change to uploaded image
   image = preprocess_image(image_path)
   prediction = model.predict(image)
   print(f'Prediction: {prediction}')
   ```

4. **Interpret Results 📈:**
   - If **sigmoid activation** is used, output is a probability (closer to 1 = Pneumonia detected).
   - If **softmax for multi-class**, specify class labels.

5. **Batch Inference (Optional) 🔄:**
   ```python
   import glob
   image_files = glob.glob('/content/drive/MyDrive/XrayDataset/*.jpg')  # Update path
   predictions = []

   for img_path in image_files:
       img = preprocess_image(img_path)
       pred = model.predict(img)
       predictions.append((img_path, pred))

   print(predictions)
   ```

6. **Visualize Predictions using Grad-CAM (Optional) 🎨:**
   ```python
   import matplotlib.pyplot as plt
   from tensorflow.keras.models import Model

   def visualize_gradcam(image_path, model):
       img = preprocess_image(image_path)
       preds = model.predict(img)
       heatmap = generate_gradcam(img, model)  # Implement Grad-CAM function separately
       plt.imshow(heatmap)
       plt.show()
   ```

## Dataset 📂
- 🏥 Trained on the **Chest X-ray dataset** obtained from [Kaggle](https://www.kaggle.com/datasets/andrewmvd/pediatric-pneumonia-chest-xray).
- 🖼️ Images are preprocessed and resized to **224x224 pixels** before training.

## Model Architecture 🏗️
- 🔬 **Base Model:** VGG16 Architecture 
- 🎯 **Attention Mechanism:** CBAM (Channel & Spatial Attention)
- 🧠 **Fully Connected Layers:** Custom classifier for Pneumonia detection

## Results 📊
- ✅ **Training Accuracy:** 91.07%
- ✅ **Validation Accuracy:** 89.33%
- ✅ **Test Accuracy:** 88.61%
- 🧮 **Loss Function:** Categorical Crossentropy
- ⚡ **Optimizer:** Adam
- 📈 **Performance Metrics Across Three Folds:**

| Metric      | Fold 1 CI         | Fold 2 CI         | Fold 3 CI         |
|------------|------------------|------------------|------------------|
| **F1-Score**  | **88.50% ± 0.60%**   | **89.10% ± 0.45%**   | **88.80% ± 0.50%**   |
| **Accuracy**  | **89.30% ± 1.10%**   | **90.10% ± 1.00%**   | **88.90% ± 1.20%**   |
| **Precision** | **88.00% ± 0.50%**   | **88.60% ± 0.40%**   | **88.40% ± 0.55%**   |
| **Recall**    | **90.30% ± 0.70%**   | **91.00% ± 0.65%**   | **90.80% ± 0.75%**   |

## License 📜
This project is licensed under the **GNU General Public License v3.0**. See the [LICENSE](LICENSE) file for details.

## Contributions Welcome 🤝
Feel free to contribute by submitting issues or pull requests. Suggestions for improvements are always welcome! ✨

## Contact 📬
For any queries, reach out via:
- 📧 **Email:** alpha992k80@gmail.com
- 🐙 **GitHub Issues:** [Open an Issue](https://github.com/Ayusharma24/CBAM-VGG16-Pneumonia-Detection/issues)

---
💡 *Let's build AI-powered healthcare solutions together!* 🚀
