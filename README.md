# MNIST Neural Network from Scratch

This project implements a fully connected neural network to classify handwritten digits from the [MNIST dataset](http://yann.lecun.com/exdb/mnist/). All operations, including data loading, matrix operations, forward propagation, backpropagation, and gradient descent, are implemented **from scratch using NumPy**, with **no deep learning libraries** like TensorFlow or PyTorch.

---

## Project Structure

nn-from-scratch/

├── load_mnist.py # Downloads and parses MNIST dataset

├── nn_scratch.py # Main training script and neural net code

├── mnist_data/ # Automatically created, stores raw MNIST files

└── README.md # You're here



---

## How to Run

1. **Clone the repo:**

```bash
git clone https://github.com/yourusername/nn-from-scratch.git
cd nn-from-scratch
``` 

2. Install Dependencies
```bash
pip install numpy matplotlib
```

3. Run the Training Script
```bash
python nn_scratch.py
```
The script will:
*Download the MNIST dataset
*Train the neural network using gradient descent
*Display predictions and images for some sample digits

## Neural Network Overview
### Architecture
* Input Layer: 784 neurons (28×28 pixels)

* Hidden Layer: 10 neurons w ReLU activation

* Output Layer: 10 neurons w softmax activation (one for each digit 0–9)

## Math Behind the Neural Network 
### Forward Propagation 
#### Hidden Layer 
``` Z[1] = W[1] * X + b[1] A[1] = ReLU(Z[1]) = max(0, Z[1]) ``` 

#### Output Layer 
``` Z[2] = W[2] * A[1] + b[2] A[2] = softmax(Z[2]) = exp(Z[2]) / sum(exp(Z[2])) ``` 

--- 
### Loss Function 
We use **categorical cross-entropy**: 
``` L = -∑ y_i * log(ŷ_i) ``` 
Where: - `y_i` is the true label (one-hot encoded) - `ŷ_i` is the predicted probability from the softmax output 

--- 
### Backward Propagation 
#### Output Layer 
``` δ[2] = A[2] - Y dW[2] = (1 / m) * δ[2] * A[1].T db[2] = (1 / m) * sum(δ[2]) ``` 

#### Hidden Layer 
``` δ[1] = (W[2].T * δ[2]) ⊙ ReLU'(Z[1]) dW[1] = (1 / m) * δ[1] * X.T db[1] = (1 / m) * sum(δ[1]) ``` > ⊙ denotes element-wise multiplication. 

--- 
### Gradient Descent 
Update Rule Each parameter is updated using: ``` W = W - α * dW b = b - α * db ``` Where `α` is the learning rate. 

