# tf-image-classification-eurosat

### README for EuroSAT Image Classification Project

#### Project Overview

This project implements a convolutional neural network (CNN) using TensorFlow for image classification into multiple categories. The goal is to develop a deep learning model capable of identifying and classifying images into various defined categories using the Eurostat dataset.

#### Dataset

The dataset used is the [EuroSAT dataset](https://www.tensorflow.org/datasets/catalog/eurosat), which is available through TensorFlow Datasets. The dataset consists of RGB satellite images covering 10 different classes, such as residential areas, forests, and farmlands.

#### Project Structure

- **Data Loading and Preparation**:
  - The project begins by loading the EuroSAT dataset using TensorFlow Datasets.
  - The data is then preprocessed and augmented to improve the model's performance and generalization capabilities.

- **Model Development**:
  - A Convolutional Neural Network (CNN) is developed using TensorFlow and Keras.
  - Various architectures and hyperparameters are explored to optimize the model.

- **Training and Evaluation**:
  - The model is trained on the preprocessed dataset, and its performance is evaluated using standard metrics.
  - Metrics such as accuracy, precision, recall, and confusion matrix are used to assess the model's performance.

#### Usage

To run the project, follow these steps:

1. **Clone the Repository**:
   ```sh
   git clone <repository_url>
   ```

2. **Install Dependencies**:
   Ensure you have TensorFlow and TensorFlow Datasets installed. You can install them using pip:
   ```sh
   pip install tensorflow tensorflow-datasets
   ```

3. **Run the Notebook**:
   Open and run the provided Jupyter notebook `tf-image-classification_eurosat.ipynb` to execute the entire workflow, from data loading to model evaluation.

#### Key Code Snippets

- **Data Loading**:
  ```python
  import tensorflow as tf
  import tensorflow_datasets as tfds

  dataset, info = tfds.load("eurosat/rgb", with_info=True, as_supervised=True)
  ```

- **Model Training**:
  ```python
  model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  history = model.fit(train_dataset, epochs=10, validation_data=validation_dataset)
  ```

- **Evaluation**:
  ```python
  test_loss, test_accuracy = model.evaluate(test_dataset)
  print(f"Test Accuracy: {test_accuracy}")
  ```

#### Resources

- [TensorFlow Datasets: EuroSAT](https://www.tensorflow.org/datasets/catalog/eurosat)
- [EuroSAT Dataset GitHub](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/image_classification/eurosat.py)

#### Contributing

Contributions to improve the project are welcome. Feel free to submit pull requests or report issues.
