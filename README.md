
# Facial Keypoints Detection Using Deep Residual Neural Networks

This project uses deep learning techniques to detect facial keypoints in grayscale images of faces. The dataset consists of 2140 images with 15 pairs of keypoints representing the coordinates of various facial features, such as the eyes, eyebrows, nose, and mouth.

### Table of Contents
* Overview
* Dataset
* Installation
* Data Preprocessing
* Model Architecture
* Training and Testing
* Augmentation Techniques
* Results Visualization

### Overview
Facial keypoint detection is an essential task in computer vision, with applications ranging from facial recognition to augmented reality. In this project, we implement a deep residual neural network to accurately identify keypoints on a face in images of size 96x96 pixels.

### Dataset
The dataset contains 2140 images of size 96x96 pixels. Each image is paired with 30 coordinates, representing 15 facial keypoints. These keypoints correspond to the following facial landmarks:

1. Eyes (center, inner corner, outer corner)
2. Eyebrows (inner and outer edges)
3. Nose tip
4. Mouth (corners, center of top lip, and bottom lip)
Each image is stored as a space-separated string of pixel values in the grayscale format.

### Installation
Ensure you have Python installed with the following packages:

* numpy
* pandas
* matplotlib
* tensorflow
* keras
* scikit-learn
To install these packages, run the following command:

    pip install numpy pandas matplotlib tensorflow keras scikit-learn

### Data Preprocessing
* **Normalization:** Image data is normalized by dividing pixel values by 255.0.
* **Image Augmentation:** To improve model generalization, several augmentation techniques are applied:
    * Horizontal and vertical flips
    * Brightness modification
* **Train-Test Split:** The dataset is split into training and testing sets using an 80-20 split.

### Model Architecture
The core of the model is a deep residual neural network implemented using Keras and TensorFlow. The residual blocks help improve gradient flow through the network, making training more effective for deep networks.

#### ResNet Block
Each residual block consists of:

- Convolutional layers with 1x1 and 3x3 filters
- Batch normalization
- Activation functions (ReLU)
- Max-pooling for downsampling
The model is constructed as follows:

- Convolutional Layer: 7x7 filters
- Residual Blocks: Multiple blocks with different filter sizes (e.g., 64, 128, 256 filters)
- Fully Connected Layer: Outputting 30 values (15 pairs of x and y coordinates)

### Training and Testing
The dataset is divided into training and testing sets. Training is performed using the following steps:

1. Compile the model: Using a mean squared error loss function and Adam optimizer.
2. Train: The model is trained on the training set and validated on the testing set.

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50)

### Augmentation Techniques
Data augmentation is crucial in this project to increase the size of the dataset and enhance the model's robustness. Augmentation methods include:

- **Horizontal and Vertical Flips:** By flipping the images, we create new variations of the dataset, improving the generalization ability of the model.
- **Brightness Adjustment:** Brightness of images is randomly increased to simulate varying lighting conditions.

### Results Visualization
After training the model, images along with their predicted keypoints can be visualized. Random samples can be displayed in a grid format using `matplotlib`:

    fig = plt.figure(figsize=(20,20))
    for i in range(16):
        ax = fig.add_subplot(4,4,i+1)
        plt.imshow(X_test[i].reshape(96,96), cmap='gray')
        plt.plot(predicted_keypoints[i, ::2], predicted_keypoints[i, 1::2], 'rx')
    
