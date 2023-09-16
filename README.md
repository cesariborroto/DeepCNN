# CIFAR-10 Image Classification

This repository contains a deep convolutional neural network (CNN) implemented using TensorFlow and Keras to classify images from the CIFAR-10 dataset. The model is trained to classify images into one of the ten categories: Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, and Truck. In addition to training the model, this repository also includes a script for making predictions on locally stored images.

## Prerequisites

Before running the code, ensure you have the following dependencies installed:

- Python 3.x
- TensorFlow 2.x
- NumPy
- Matplotlib
- Pillow (PIL)

You can install these dependencies using `pip`:

```bash
pip install tensorflow numpy matplotlib Pillow
```

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/username/repo.git
   cd repo
   ```

2. Download the CIFAR-10 dataset:
   
   The dataset will be automatically downloaded when you run the script for the first time.

3. Train the model:

   Run the following command to train the CNN model on the CIFAR-10 dataset:

   ```bash
   python train.py
   ```

   The model will be trained for 15 epochs, and the training progress will be displayed in the console.

4. Make predictions on a local image:

   To make predictions on an image stored on your local machine, replace `'image.jpg'` with the path to your image in the `image_path` variable inside the `predict.py` script. Then, run the following command:

   ```bash
   python predict.py
   ```

   The script will load and preprocess the image, make predictions, and display the predicted class along with the image.

## Model Architecture

The CNN model used in this repository has the following architecture:

1. Input layer: 32x32x3 (RGB image)
2. Convolutional layer with 32 filters, 3x3 kernel, and ReLU activation
3. Max-pooling layer (2x2)
4. Convolutional layer with 64 filters, 3x3 kernel, and ReLU activation
5. Max-pooling layer (2x2)
6. Convolutional layer with 64 filters, 3x3 kernel, and ReLU activation
7. Flatten layer
8. Fully connected layer with 64 units and ReLU activation
9. Fully connected layer with 10 units and softmax activation

The model is compiled using the Adam optimizer and the sparse categorical cross-entropy loss function.

## Results

After training the model, you can visualize the training and validation accuracy by running:

```bash
python plot_accuracy.py
```

This will display a plot showing the training and validation accuracy over the epochs.

## Author

- [Cesar Borroto](https://github.com/cesarborroto)
