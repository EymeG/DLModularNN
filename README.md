# Project Overview

This project implements a modular neural network building approach, enabling the construction of neural networks with customizable architectures. The project supports both regression and multilabel classification tasks.

## Modular Neural Network Building

The project follows a modular architecture inspired by Keras, allowing users to construct neural networks by adding layers of different types, sizes, and activation functions. This modular approach provides flexibility and scalability in designing neural network architectures tailored to specific tasks.

## Regression and Multiclass classification

### Regression

For regression tasks, the neural network can predict continuous numerical values. It's suitable for applications such as predicting house prices, stock prices, or any other continuous variable.

### Multilabels classification

For multilabels classification tasks, the neural network can classify input data into multiple classes. It's commonly used in applications like image classification, where the neural network assigns an input image to one of several predefined categories.

### `NeuralNetwork`

The `NeuralNetwork` class represents a neural network model in which the user can add Layer by himself, following the modular approach of python Keras library.

#### Constructor:

- **`NeuralNetwork(double lr, int nEpochs, std::string lossFunction)`**: Constructs a neural network with the specified learning rate, number of epochs to perform during training, and the loss function to use.

#### Methods:

- **`addLayer(const Layer& l)`**: Adds a layer to the neural network.
- **`feedForward(int inputIndex, bool isTraining)`**: Performs forward propagation on the neural network.
- **`backPropagation(int inputIndex)`**: Performs backpropagation to update weights and biases.
- **`train(const std::vector<std::vector<double>>& inputs, const std::vector<double>& expectedOutputs)`**: Trains the neural network using the specified input and output data.
- **`evaluate(std::vector<std::vector<double>> entry)`**: Evaluates the neural network on the given input data.
- **`save(std::string filename)`**: Saves the trained model to a file.
- **`load(const std::string& filename)`**: Loads a trained model from a file.
- **`summary()`**: Displays a summary of the neural network configuration and parameters.
- **`confusion(std::vector<std::vector<double>> inputs, std::vector<double> expectedOutputs)`**: Computes and displays the confusion matrix for evaluating the model's performance of a multilabel classification model(accuracy & precision).

#### Destructor:

- **`~NeuralNetwork()`**: Default destructor.

---

### `Layer`

The `Layer` class represents a single layer in a neural network.

#### Constructors:

- **`Layer(int layerSize, const std::string& activation)`**: Constructs a layer with the specified size and activation function.

#### Methods:

- **`generateRandomVector(int size) -> std::vector<double>`**: Generates a random vector of the specified size.
- **`initWeights()`**: Initializes the layer weights with random values.
- **`initBiases()`**: Initializes the layer biases with random values.
- **`setInputs(const std::vector<double>& inputs)`**: Sets the input values for the layer.
- **`computeOutputs(bool isInputLayer)`**: Computes the outputs of the layer.
- **`getLayerDerivative() -> std::vector<double>`**: Computes and returns the derivative of the layer.
- **`getLossDerivative(const double& expectedOutput, const std::string& lossFunction) -> std::vector<double>`**: Computes and returns the derivative of the loss function.
- **`computeActivation()`**: Computes the activation function for the layer outputs.
- **`updateLayerWeightsAndBiases(const std::vector<double>& deltas, double learningRate)`**: Updates the layer weights and biases using backpropagation.
- **`getOutputs() -> std::vector<double>`**: Returns the outputs of the layer.
- **`getWeights() -> std::vector<double>`**: Returns the weights of the layer.
- **`getPostactivation() -> std::vector<double>`**: Returns the post-activation outputs of the layer.
- **`setBiases(const std::vector<double>& b)`**: Sets the biases of the layer.
- **`setWeights(const std::vector<double>& w)`**: Sets the weights of the layer.
- **`setOutputs(const std::vector<double>& o)`**: Sets the outputs of the layer.
- **`getDeltas(const std::vector<double>& lastdeltas, Layer& nextLayer) -> std::vector<double>`**: Computes and returns the deltas for backpropagation.
- **`getLoss(double expectedOutput, const std::string& lossFunction) -> double`**: Computes and returns the loss for the layer.
- **`setLastLayerSize(int i)`**: Sets the size of the last layer.
- **`initLayer()`**: Initializes the layer with random weights and biases.
- **`saveLayer(std::ofstream& file)`**: Saves the layer parameters to a file.

#### Additional Constructors:

- **`Layer()`**: Default constructor.

#### Helper Methods:

- **`layerSummary(int i) -> int`**: Prints a summary of the layer configuration and returns the number of parameters.

---


## Usage Example: 

### Regression on XOR Function

In this example, we perform regression using a neural network to approximate the XOR function. I used BinaryCrossEntropy loss and MSE loss for this example.

The example can be find in the file 
```file 
mainXOR.cpp
```

### Multilabel classification:

In this example, we perform the classification on written number from the MNIST dataset.

The example can be find in the file
```
mainMNIST.cpp
```

You may need to download the MNIST dataset and to change the path to the file in the code to make it work.
