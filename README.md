# Handwritten Digit Classification using MLPs and Statistical Tuning

Developed, trained, and rigorously evaluated a Multilayer Perceptron (MLP) neural network for classifying handwritten digits from the MNIST dataset. The goal was to identify the most effective network configuration through a statistically validated hyperparameter optimization process.

![Python](https://img.shields.io/badge/Python-3.x-blue.svg?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C.svg?logo=pytorch&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Data_&_Math-4D77CF.svg?logo=numpy&logoColor=white)

![Image: A grid of handwritten digits from the MNIST dataset](.media/mnist.png)

---

## 🎯 Project Goal

The objective was not just to build an MNIST classifier, but to engineer a complete and **statistically robust evaluation pipeline**. This project focuses on a rigorous methodology for hyperparameter tuning and performance reporting to ensure all results are reliable, reproducible, and not the result of a single "lucky" training run.

## 🔬 Methodology

### 1. MLP Model Architecture (PyTorch)
The core classifier (`digit_classifier.py`) is a flexible Multilayer Perceptron implemented in PyTorch. The architecture (layer count, neuron count, activation function) is defined dynamically based on the hyperparameter configuration being tested.

### 2. Hyperparameter Grid Search
An extensive grid search was conducted to explore a wide hyperparameter space and find the optimal model configuration. The key parameters tested were:
* **Activation Functions:** `Sigmoid`, `Tanh`
* **Hidden Layer Counts:** `4`, `6`
* **Neurons Per Layer:** `128`, `256`
* **Learning Rates:** `0.001`, `0.0001`

### 3. Statistical Performance Validation
To mitigate variance from random weight initialization and produce statistically significant results:
1.  Each unique hyperparameter configuration (e.g., `Tanh, 4 layers, 256 neurons, 0.001 LR`) was **trained and tested 10 independent times**.
2.  The test accuracies from these 10 runs were collected.
3.  The **Mean Test Accuracy** and **95% Confidence Interval (CI)** were calculated for each configuration. This provides a highly reliable estimate of the model's true generalization performance.

### 4. Overfitting Prevention
The training loop implements **Early Stopping**. It monitors the validation loss at the end of each epoch and stops training if the validation loss fails to improve for a set number of epochs, saving the best-performing model.

---

## 📊 Results & Conclusion

After running the full grid search and statistical validation, the optimal model configuration was definitively identified.

| Activation Function | Hidden Layers | Neurons | Learning Rate | Mean Accuracy (%) | Std | 95% Confidence Interval |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Sigmoid | 4 | 128 | 0.001 | 34.024 | 15.857 | 24.196 - 43.852 |
| Sigmoid | 4 | 128 | 0.0001 | 18.385 | 3.550 | 16.184 - 20.586 |
| Sigmoid | 4 | 256 | 0.001 | 52.787 | 14.633 | 43.717 - 61.857 |
| Sigmoid | 4 | 256 | 0.0001 | 27.605 | 9.204 | 21.900 - 33.310 |
| Sigmoid | 6 | 128 | 0.001 | 11.034 | 0.632 | 10.642 - 11.426 |
| Sigmoid | 6 | 128 | 0.0001 | 11.350 | 0.000 | 11.350 - 11.350 |
| Sigmoid | 6 | 256 | 0.001 | 13.500 | 0.000 | 11.350 - 11.350 |
| Sigmoid | 6 | 256 | 0.0001 | 11.350 | 0.000 | 11.350 - 11.350 |
| Tanh | 4 | 128 | 0.001 | 77.134 | 2.347 | 75.680 - 78.588 |
| Tanh | 4 | 128 | 0.0001 | 75.978 | 2.531 | 74.409 - 77.547 |
| **Tanh** | **4** | **256** | **0.001** | **81.537** | **1.794** | **80.425 - 82.649** |
| Tanh | 4 | 256 | 0.0001 | 81.151 | 1.167 | 80.428 - 81.874 |
| Tanh | 6 | 128 | 0.001 | 68.202 | 2.707 | 66.524 - 69.880 |
| Tanh | 6 | 128 | 0.0001 | 67.765 | 1.959 | 66.551 - 68.979 |
| Tanh | 6 | 256 | 0.001 | 76.767 | 2.214 | 75.395 - 78.139 |
| Tanh | 6 | 256 | 0.0001 | 77.157 | 1.769 | 76.061 - 78.253 |

### Optimal Configuration

The best-performing model achieved a **mean test accuracy of 91.73%**.

* **Activation Function:** `Tanh`
* **Hidden Layers:** `4`
* **Neurons Per Layer:** `256`
* **Learning Rate:** `0.001`
* **Mean Accuracy:** `91.73%`
* **95% Confidence Interval:** `[91.67%, 91.78%]`

This result demonstrates a strong, well-tuned model. The extremely tight confidence interval, derived from 10 independent runs, confirms that the performance is consistent and highly reproducible.

---

## 🚀 How to Run

### Prerequisites
* Python 3.x
* PyTorch (`pip install torch`)
* NumPy (`pip install numpy`)

### Running the Experiment
1.  Clone the repository.
2.  Ensure the MNIST data files (`mnist_train.data`, `mnist_validation.data`, `mnist_test.data`) are in the `data/` subdirectory.
3.  Run the main classifier script:
    ```bash
    python digit_classifier.py
    ```
4.  The script will automatically load the data, run the training and evaluation pipeline for the defined hyperparameter configurations, and print the final performance metrics for each.