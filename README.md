# Project Overview

This project implements a modular neural network building approach, enabling the construction of neural networks with customizable architectures. The project supports both regression and multilabel classification tasks.

## Modular Neural Network Building

The project follows a modular architecture inspired by Keras, allowing users to construct neural networks by adding layers of different types, sizes, and activation functions. This modular approach provides flexibility and scalability in designing neural network architectures tailored to specific tasks.

## Features
- Build a new model from scratch for classifications and regression. You can set the learning rate and the number of epochs to perform during training.
- Choose between 3 loss methods (Regression: "mse", Classification: "bin_crossentropy" & "cat_crossentropy").
- Add custom layers to the model. You can customize layer sizes and activation functions ("sigmoid", "linear", "tanh", "relu", "softmax").
- Train the NN on a custom dataset by using `train` with a `std::vector<std::vector<double>>` as inputs, and `std::vector<double>` as expected outputs.
- Save the model in .txt format after training and load it later using the `save` and `load` methods.
- Evaluate the model after training using the `evaluate` method with input `std::vector<std::vector<double>>`, the set of vectors you want to evaluate using the model.
- Print performance summary using the `confusion()` method on a multilabel classification model (softmax output activation).

## Commentaries:
- For multilabel classification, one-hot-encoding is included inside the implementation. For a three-class model, for example, expected output must be indicated using their class name from 0 to 2 (0 for class 1, 1 for class 2, 2 for class 3).
- The `confusion()` method can only be used on a multilabel classification model (softmax output activation).
- Data formatting has to be done manually by the user depending on their dataset. Examples of data formatting are given in the `main.cpp` files for MNIST classification, XOR regression, and Sentiment Analysis examples.

## Usage Example:

## Project Usage:

To use this project, download this GitHub repository and start it using CMake (You can use CLion to do it, for example). The examples that follow can only work if you copy the cmake_debug file, in which the datasets are stored for those examples. For each example, you have to change the name of the main.cpp file in the `CMakeLists.txt` file to the correct one to launch the desired main file.

### Regression on XOR Function

In this example, we perform regression by training a neural network to approximate the XOR function.

The example can be found in the file:
```
mainXOR.cpp
```

### Multilabel Classification:

In this example, we perform the classification on handwritten numbers from the MNIST dataset.

The example can be found in the file:
```
mainMNIST.cpp
```

You may need to download the MNIST dataset and change the path to the file in the code to make it work.

### Loading Model Examples:

In this example, we are loading a model for XOR regression and evaluating it on the different XOR possible entries.

The example can be found in the file:
```
mainXORloadtest.cpp
```

### Sentiment Analysis:

In this example, we try to perform a multilabel classification task on tweets. This example is not fully functional due to the optimization method used to update the weights of the model (SGD without momentum). I leave the example as an example of data formatting.

This example can be found in the file:
```
mainSentiment.cpp
```

### `NeuralNetwork`

The `NeuralNetwork` class represents a neural network model in which the user can add layers by themselves, following the modular approach of the Python Keras library.

#### Constructor:

- **`NeuralNetwork(double lr, int nEpochs, std::string lossFunction)`**: Constructs a neural network with the specified learning rate, number of epochs to perform during training, and the loss function to use.

#### Methods:

- **`public addLayer(const Layer& l)`**: Adds a layer to the neural network.
- **`private feedForward(int inputIndex, bool isTraining)`**: Performs forward propagation on the neural network.
- **`private backPropagation(int inputIndex)`**: Performs backpropagation to update weights and biases.
- **`public train(const std::vector<std::vector<double>>& inputs, const std::vector<double>& expectedOutputs)`**: Trains the neural network using the specified input and output data.
- **`public evaluate(std::vector<std::vector<double>> entry)`**: Evaluates the neural network on the given input data.
- **`public save(std::string filename)`**: Saves the trained model to a file.
- **`public load(const std::string& filename)`**: Loads a trained model from a file.
- **`public summary()`**: Displays a summary of the neural network configuration and parameters.
- **`public confusion(std::vector<std::vector<double>> inputs, std::vector<double> expectedOutputs)`**: Computes and displays the confusion matrix for evaluating the model's performance of a multilabel classification model (softmax output activation) (accuracy, precision, and recall).
- **`public ~NeuralNetwork()`**: Default destructor.

---

### `Layer`

The `Layer` class represents a single layer in a neural network.

#### Constructors:

- **`Layer(int layerSize, const std::string& activation)`**: Constructs a layer with the specified size and activation function.

#### Methods:

- **`private generateRandomVector(int size) -> std::vector<double>`**: Generates a random vector of the specified size.
- **`public initWeights()`**: Initializes the layer weights with random values.
- **`public initBiases()`**: Initializes the layer biases with random values.
- **`public setInputs(const std::vector<double>& inputs)`**: Sets the input values for the layer.
- **`public computeOutputs(bool isInputLayer)`**: Computes the outputs of the layer.
- **`public getLayerDerivative() -> std::vector<double>`**: Computes and returns the derivative of the layer.
- **`public getLossDerivative(const double& expectedOutput, const std::string& lossFunction) -> std::vector<double>`**: Computes and returns the derivative of the loss function.
- **`public computeActivation()`**: Computes the activation function for the layer outputs.
- **`public updateLayerWeightsAndBiases(const std::vector<double>& deltas, double learningRate)`**: Updates the layer weights and biases using backpropagation.
- **`public getOutputs() -> std::vector<double>`**: Returns the outputs of the layer.
- **`public getWeights() -> std::vector<double>`**: Returns the weights of the layer.
- **`public getPostactivation() -> std::vector<double>`**: Returns the post-activation outputs of the layer.
- **`public setBiases(const std::vector<double>& b)`**: Sets the biases of the layer.
- **`public setWeights(const std::vector<double>& w)`**: Sets the weights of the layer.
- **`public setOutputs(const std::vector<double>& o)`**: Sets the outputs of the layer.
- **`public getDeltas(const std::vector<double>& lastdeltas, Layer& nextLayer) -> std::vector<double>`**: Computes and returns the deltas for backpropagation.
- **`public getLoss(double expectedOutput, const std::string& lossFunction) -> double`**: Computes and returns the loss for the layer.
- **`public setLastLayerSize(int i)`**: Sets the size of the last layer.
- **`public initLayer()`**: Initializes the layer with random weights and biases.
- **`public saveLayer(std::ofstream& file)`**: Saves the layer parameters to a file.
- **`private loadLayer(std::ifstream& file)`**: Loads a layer from a file.

#### Additional Constructors:

- **`Layer()`**: Default constructor.

#### Helper Methods:

- **`layerSummary(int i) -> int`**: Prints a summary of the layer configuration and returns the number of parameters.

---



# Author
Project developped by Eymeric GIABICANI.
