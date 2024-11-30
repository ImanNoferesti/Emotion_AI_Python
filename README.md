
# Emotion and Facial Key-Point Detection

This project aims to classify people’s emotions based on their facial images and predict key facial points using deep learning models. It involves training and deploying systems to monitor facial emotions and expressions automatically.

## Project Overview

### Part 1: Key Facial Points Detection
* **Objective:** Predict the x and y coordinates of 15 key facial points.
* **Dataset:** Includes over 2000 images with facial key-point annotations.
* **Approach:** Build a Convolutional Neural Network (CNN) with Residual Blocks (ResNet) to detect key facial points.

### Part 2: Facial Expression (Emotion) Detection
* **Objective:** Classify facial emotions into 5 categories:
  * 0: Angry 😡
  * 1: Disgust 😖
  * 2: Sad 😔
  * 3: Happy 😄
  * 4: Surprise 😲
* **Dataset:** Contains more than 20,000 labeled facial images.

## Project Steps
**General Steps**
1. **Image Visualization:** Explore and understand the dataset through visualizations.
2. **Image Augmentation:** Apply transformations to enhance the dataset.
3. **Data Normalization and Scaling:** Prepare data for efficient model training.

**Key Facial Points Detection**

4. **Build ResNet Model:** Design a CNN with Residual Blocks for key point detection.
5. **Compile and Train Model:** Train the model to predict facial key points.
6. **Model Performance Assessment:** Evaluate the model’s accuracy and reliability.

**Facial Expression Detection**

7. **Explore Emotion Dataset:** Import and analyze the emotion dataset.
8. **Visualize Emotion Data:** Gain insights through visual exploration.
9. **Build Emotion Classifier:** Create a classifier model for emotion detection.
10. **Train the Model:** Optimize the classifier to detect emotions accurately.
11. **Evaluate Classifier:** Understand key performance indicators (KPIs) and assess results.

**Predictions and Deployment**

12. **Model Predictions:** Generate predictions using:
   * Key Facial Points Detection Model
   * Emotion Classifier Model
13. **Save Trained Models:** Prepare models for deployment.
14. **TensorFlow Serving:** Serve models using TensorFlow Serving.
        
