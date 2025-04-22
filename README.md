![sdsffsdf.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/RZ0c2zsJESPWMy4avejVY.png)

# **Rice-Leaf-Disease** 🌾  

> **Rice-Leaf-Disease** is an image classification model fine-tuned from **google/siglip2-base-patch16-224** for detecting and categorizing diseases in rice leaves. It is built using the **SiglipForImageClassification** architecture and helps in early identification of plant diseases for better crop management.
> 
```py
Classification Report:
                 precision    recall  f1-score   support

Bacterialblight     0.8853    0.9596    0.9210      1585
          Blast     0.9271    0.8472    0.8853      1440
      Brownspot     0.9746    0.9369    0.9554      1600
        Healthy     1.0000    1.0000    1.0000      1488
         Tungro     0.9589    0.9977    0.9779      1308

       accuracy                         0.9477      7421
      macro avg     0.9492    0.9483    0.9479      7421
   weighted avg     0.9486    0.9477    0.9474      7421
```

![download (1).png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/iuXCriQpPXJmLeMy--WJr.png)

### **Disease Categories:**  
- **Class 0:** Bacterial Blight  
- **Class 1:** Blast  
- **Class 2:** Brown Spot  
- **Class 3:** Healthy  
- **Class 4:** Tungro  

---

# **Run with Transformers 🤗**  

```python
!pip install -q transformers torch pillow gradio
```  

```python
import gradio as gr
from transformers import AutoImageProcessor, SiglipForImageClassification
from transformers.image_utils import load_image
from PIL import Image
import torch

# Load model and processor
model_name = "prithivMLmods/Rice-Leaf-Disease"
model = SiglipForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

def classify_leaf_disease(image):
    """Predicts the disease type in a rice leaf image."""
    image = Image.fromarray(image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()
    
    labels = {
        "0": "Bacterial Blight",
        "1": "Blast",
        "2": "Brown Spot",
        "3": "Healthy",
        "4": "Tungro"
    }
    predictions = {labels[str(i)]: round(probs[i], 3) for i in range(len(probs))}
    
    return predictions

# Create Gradio interface
iface = gr.Interface(
    fn=classify_leaf_disease,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(label="Prediction Scores"),
    title="Rice Leaf Disease Classification 🌾",
    description="Upload an image of a rice leaf to identify if it is healthy or affected by diseases like Bacterial Blight, Blast, Brown Spot, or Tungro."
)

# Launch the app
if __name__ == "__main__":
    iface.launch()
```

---

# **Intended Use:**  

The **Rice-Leaf-Disease** model helps in detecting and classifying rice leaf diseases early, supporting:  
✅ **Farmers & Agriculturists:** Quick disease detection for better crop management.  
✅ **Agricultural Research:** Monitoring and analyzing plant disease patterns.  
✅ **AI & Machine Learning Projects:** Applying AI to real-world agricultural challenges.  
