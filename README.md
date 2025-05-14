# 🎨 Image Insight Analyzer

A web application that analyzes images by detecting their **dominant colors**, suggesting a **mood** based on those colors, and predicting the **scene** shown in the image using OpenAI’s CLIP model.

---

## 🚀 Features

- Extracts 5 dominant colors from any uploaded image
- Matches colors to human-readable names (e.g., `teal`, `saddlebrown`)
- Suggests a mood (e.g., Calm, Intense, Fresh) based on color palette
- Uses CLIP for high-level scene classification (e.g., beach, food, portrait)
- Clean and simple web interface (built with Flask + HTML/CSS)

---

## 📸 How It Works

Here's how the application processes your image step by step:

1. **Upload an Image**  
   The user uploads an image through a simple browser-based form.

2. **Color Extraction**  
   The system resizes the image and applies KMeans clustering to identify the top 5 dominant colors with high saturation and brightness.

3. **Color Naming and Mood Inference**  
   Each color is matched to a human-readable name (e.g., `teal`, `maroon`). Then, a rule-based logic analyzes the palette to suggest an overall mood such as “Fresh / Natural” or “Intense / Passionate”.

4. **Scene Prediction**  
   Using OpenAI’s CLIP model, the image is semantically compared to a predefined list of possible scenes (like “a beach”, “a portrait”, etc.). The most likely match is selected as the scene classification.

5. **Display Results**  
   The browser interface displays the original image, the color palette, their names and RGB values, the inferred mood, and the predicted scene—all in an organized and readable layout.


