//
// Created by eymeg on 04/06/2024.
//

#include <random>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <sstream>
#include "DenseLayer.h"

// Generates a random vector of specified size with values in the range [-1, 1]
std::vector<double> Layer::generateRandomVector(int size) {
    std::vector<double> res(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> d(-1, 1);
    for (int i = 0; i < size; ++i) {
        res[i] = d(gen);
    }
    return res;
}

// Initializes the weights of the layer with random values
void Layer::initWeights() {
    weights = generateRandomVector(layerSize * lastLayerSize);
}

// Initializes the biases of the layer with random values
void Layer::initBiases() {
    biases = generateRandomVector(layerSize);
}

// Sets the input values for the layer
// Parameters:
// - inputs: a vector of input values to be set for the layer
void Layer::setInputs(const std::vector<double> &inputs) {
    this->input = inputs;
}

// Constructor to initialize the layer with its size and activation function
// Parameters:
// - layerSize: the number of neurons in the layer
// - activation: the activation function to be used in the layer
Layer::Layer(int layerSize, const std::string &activation) {
    this->layerSize = layerSize;
    this->activation = activation;
}

// Returns the size of the layer
int Layer::getLayerSize() const {
    return this->layerSize;
}

// Computes the outputs of the layer
// Parameters:
// - isInputLayer: a boolean indicating if the layer is an input layer
void Layer::computeOutputs(bool isInputLayer) {
    if (isInputLayer) {
        outputs = input;
    } else {
        std::vector<double> res(layerSize);
        for (int i = 0; i < layerSize; ++i) {
            double sum = 0;
            for (int j = 0; j < lastLayerSize; ++j) {
                sum += weights[j + i * lastLayerSize] * input[j];
            }
            res[i] = sum + biases[i];
        }
        outputs = res;
    }
}

// Calculates the derivative of the activation function for the layer
std::vector<double> Layer::getLayerDerivative() {
    std::vector<double> res(layerSize);
    for (int i = 0; i < layerSize; ++i) {
        if (activation == "input") {
            res[i] = 1;
        } else if (activation == "relu") {
            res[i] = outputs[i] > 0 ? 1 : 0;
        } else if (activation == "sigmoid") {
            res[i] = postactivation[i] * (1 - postactivation[i]);
        } else if (activation == "softmax") {
            res[i] = postactivation[i] * (1 - postactivation[i]);
        } else if (activation == "tanh") {
            res[i] = 1 - postactivation[i] * postactivation[i];
        } else if (activation == "linear") {
            res[i] = 1;
        } else {
            throw std::invalid_argument("Invalid activation function");
        }
    }
    return res;
}

// Calculates the derivative of the loss function with respect to the outputs
// Parameters:
// - expectedOutput: the expected output value
// - lossFunction: the loss function to be used ("bin_crossentropy", "mse", "cat_crossentropy")
std::vector<double> Layer::getLossDerivative(const double& expectedOutput, const std::string &lossFunction) {
    std::vector<double> res(layerSize);
    if (lossFunction == "bin_crossentropy") {
        for (int i = 0; i < layerSize; ++i) {
            double espilon = 1e-10;
            if (postactivation[i] == 0) {
                res[i] = (1 - expectedOutput) - expectedOutput/espilon;
            } else if (postactivation[i] == 1) {
                res[i] = (1 - expectedOutput)/(1-espilon) - expectedOutput;
            } else {
                res[i] = (1 - expectedOutput)/(1-postactivation[i]) - expectedOutput/postactivation[i];
            }
        }
    } else if (lossFunction == "mse") {
        for (int i = 0; i < layerSize; ++i) {
            res[i] = postactivation[i] - expectedOutput;
        }
    } else if (lossFunction == "cat_crossentropy") {
        for (int i = 0; i < layerSize; ++i) {
            if (expectedOutput == i) {
                res[i] = postactivation[i] - 1;
            } else {
                res[i] = postactivation[i];
            }
        }
    } else {
        throw std::invalid_argument("Invalid loss function");
    }
    return res;
}

// Applies the activation function to the computed outputs
void Layer::computeActivation() {
    std::vector<double> res(layerSize);
    if (activation == "input") {
        res = outputs;
    }
    for (int i = 0; i < layerSize; ++i) {
        if (activation == "relu") {
            res[i] = outputs[i] > 0 ? outputs[i] : 0;
        } else if (activation == "sigmoid") {
            res[i] = 1 / (1 + std::exp(-outputs[i]));
        } else if (activation == "tanh") {
            res[i] = std::tanh(outputs[i]);
        } else if (activation == "linear") {
            res[i] = outputs[i];
        }
    }
    if (activation == "softmax") {
        double sum = 0;
        for (int i = 0; i < layerSize; ++i) {
            sum += exp(outputs[i]);
        }
        for (int i = 0; i < layerSize; ++i) {
            res[i] = exp(outputs[i]) / sum;
        }
    }
    postactivation = res;
}

// Updates the weights and biases of the layer based on the computed deltas and learning rate
// Parameters:
// - deltas: a vector of deltas to update the weights and biases
// - learningRate: the learning rate for the update
void Layer::updateLayerWeightsAndBiases(const std::vector<double> &deltas, double learningRate) {
    for (int i = 0; i < layerSize; ++i) {
        for (int j = 0; j < lastLayerSize; ++j) {
            weights[j + i * lastLayerSize] -= learningRate * deltas[i] * input[j];
        }
    }
    for (int i = 0; i < layerSize; ++i) {
        biases[i] -= learningRate * deltas[i];
    }
}

// Returns the outputs of the layer
std::vector<double> Layer::getOutputs() const {
    return outputs;
}

// Returns the weights of the layer
std::vector<double> Layer::getWeights() const {
    return weights;
}

// Returns the post-activation outputs of the layer
std::vector<double> Layer::getPostactivation() const {
    return postactivation;
}

// Sets the biases of the layer
// Parameters:
// - b: a vector of biases to be set for the layer
void Layer::setBiases(const std::vector<double> &b) {
    this->biases = b;
}

// Sets the weights of the layer
// Parameters:
// - w: a vector of weights to be set for the layer
void Layer::setWeights(const std::vector<double> &w) {
    this->weights = w;
}

// Sets the outputs of the layer
// Parameters:
// - o: a vector of outputs to be set for the layer
void Layer::setOutputs(const std::vector<double> &o) {
    this->outputs = o;
}

// Calculates the deltas for the layer based on the deltas from the next layer
// Parameters:
// - lastdeltas: a vector of deltas from the next layer
// - nextLayer: the next layer object
std::vector<double> Layer::getDeltas(const std::vector<double> &lastdeltas, Layer &nextLayer) {
    std::vector<double> res(layerSize);
    std::vector<double> layerDerivative = getLayerDerivative();
    for (int i = 0; i < layerSize; ++i) {
        double sum = 0;
        for (int j = 0; j < nextLayer.getLayerSize(); ++j) {
            sum += nextLayer.getWeights()[j] * lastdeltas[j];
        }
        res[i] = sum * layerDerivative[i];
    }
    return res;
}

// Computes the loss of the layer
// Parameters:
// - expectedOutput: the expected output value
// - lossFunction: the loss function to be used ("bin_crossentropy", "mse", "cat_crossentropy")
double Layer::getLoss(double expectedOutput, const std::string &lossFunction) {
    if (lossFunction == "bin_crossentropy") {
        double predictvalue = postactivation[0];
        return -expectedOutput * (std::log(predictvalue)) - (1 - expectedOutput) * (std::log(1 - predictvalue));
    }
    if (lossFunction == "mse") {
        return (postactivation[0] - expectedOutput) * (postactivation[0] - expectedOutput);
    } if (lossFunction == "cat_crossentropy") {
        double sum = 0;
        for (int i = 0; i < layerSize; ++i) {
            if (expectedOutput == i) {
                sum += 1 * std::log(postactivation[i]);
            }
        }
        return -sum;
    }
    return 0;
}

// Sets the size of the last layer
// Parameters:
// - i: the size of the last layer
void Layer::setLastLayerSize(int i) {
    this->lastLayerSize = i;
}

// Initializes the layer by initializing its weights and biases
void Layer::initLayer() {
    initWeights();
    initBiases();
}

// Saves the layer configuration to a file
// Parameters:
// - file: an ofstream object representing the file to save to
void Layer::saveLayer(std::ofstream& file) {
    if (!file.is_open()) {
        std::cerr << "Erreur : le fichier n'est pas ouvert." << std::endl;
        return;
    }
    file << "#" << std::endl;
    file << layerSize << " " << activation << " " << lastLayerSize << std::endl;

    for (int i = 0; i < layerSize; ++i) {
        file << biases[i] << " ";
    }
    file << std::endl;

    for (int i = 0; i < layerSize; ++i) {
        for (int j = 0; j < lastLayerSize; ++j) {
            file << weights[j + i * lastLayerSize] << " ";
        }
    }
    file << std::endl;
}

// Default constructor for Layer
Layer::Layer() {
    this->layerSize = 0;
    this->activation = "input";
    this->lastLayerSize = 0;
    this->input = std::vector<double>();
    this->weights = std::vector<double>();
    this->biases = std::vector<double>();
    this->outputs = std::vector<double>();
    this->postactivation = std::vector<double>();
}

// Prints a summary of the layer configuration and returns the number of parameters
// Parameters:
// - i: the index of the layer
int Layer::layerSummary(int i) {
    if (activation != "input") {
        std::cout << "Layer " << i << " | Size: " << layerSize << " | Activation: " << activation << " | Params: " << layerSize * (lastLayerSize + 1) << std::endl;
    } else {
        std::cout << "Layer " << i << " | Size: " << layerSize << " | Activation: " << activation << std::endl;
    }
    std::cout << "------------------------------------------------" << std::endl;

    return layerSize * (lastLayerSize + 1);
}
