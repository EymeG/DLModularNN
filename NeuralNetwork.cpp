#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <iomanip>
#include "NeuralNetwork.h"

// Constructor to initialize the neural network with learning rate, number of epochs, and loss function
// Parameters:
// - lr: learning rate
// - nEpochs: number of training epochs
// - lossFunction: loss function to use ("mse", "bin_crossentropy", etc.)
NeuralNetwork::NeuralNetwork(double lr, int nEpochs, std::string lossFunction) {
    this->learningRate = lr;
    this->nEpochs = nEpochs;
    this->nLayers = 0;  // Initialize nLayers
    this->totalloss = 0.0;  // Initialize totalloss
    this->lossFunction = lossFunction;
}

// Adds a layer to the neural network
// Parameters:
// - l: the layer to be added
void NeuralNetwork::addLayer(const Layer& l) {
    layers.push_back(l);
    nLayers++;
    layers[nLayers - 1].setLastLayerSize(nLayers > 1 ? layers[nLayers - 2].getLayerSize() : 0);
    layers[nLayers - 1].initLayer();
}

// Performs feedforward operation on the network
// Parameters:
// - inputIndex: index of the input vector to use
// - isTraining: boolean flag indicating if the operation is for training
void NeuralNetwork::feedForward(int inputIndex, bool isTraining) {
    std::vector<double> input = inputs[inputIndex];
    if (input.size() != layers[0].getLayerSize()){
        throw std::invalid_argument("Input size does not match input layer size");
    }
    layers[0].setInputs(input);
    layers[0].computeOutputs(true);
    layers[0].computeActivation();
    for (int i = 1; i < nLayers; i++) {
        layers[i].setInputs(layers[i - 1].getPostactivation());
        layers[i].computeOutputs(false);
        layers[i].computeActivation();
    }
    if (isTraining) {
        totalloss += layers[nLayers - 1].getLoss(expectedOutputs[inputIndex], lossFunction);
    }
}

// Performs backpropagation to update weights and biases
// Parameters:
// - inputIndex: index of the input vector to use
void NeuralNetwork::backPropagation(int inputIndex) {
    std::vector<std::vector<double>> Deltas(nLayers);
    std::vector<double> thisLayerDeltas;
    std::vector<double> thisLayerDerivative = layers[nLayers - 1].getLayerDerivative();
    std::vector<double> thisLossDerivative = layers[nLayers - 1].getLossDerivative(expectedOutputs[inputIndex], lossFunction);

    // Compute delta for the output layer
    thisLayerDeltas.reserve(thisLayerDerivative.size());
    for (int i = 0; i < thisLayerDerivative.size(); i++) {
        thisLayerDeltas.push_back(thisLayerDerivative[i] * thisLossDerivative[i]);
    }
    Deltas[nLayers - 1] = thisLayerDeltas;

    // Compute delta for hidden layers
    for (int i = nLayers - 2; i >= 0; i--) {
        thisLayerDeltas.clear();
        std::vector<double> nextLayerWeights = layers[i + 1].getWeights();
        for (int j = 0; j < layers[i].getLayerSize(); j++) {
            double delta = 0.0;
            for (int k = 0; k < layers[i + 1].getLayerSize(); k++) {
                delta += nextLayerWeights[k * layers[i].getLayerSize() + j] * Deltas[i + 1][k];
            }
            delta *= layers[i].getLayerDerivative()[j];
            thisLayerDeltas.push_back(delta);
        }
        Deltas[i] = thisLayerDeltas;
    }

    // Update weights and biases for all layers
    for (int i = 0; i < nLayers; i++) {
        layers[i].updateLayerWeightsAndBiases(Deltas[i], learningRate);
    }
}

// Trains the neural network on the provided inputs and expected outputs
// Parameters:
// - inputs: a vector of input vectors
// - expectedOutputs: a vector of expected output values
void NeuralNetwork::train(const std::vector<std::vector<double>> &inputs, const std::vector<double> &expectedOutputs) {
    std::cout << "Training begin: " << std::endl;
    this->inputs = inputs;
    this->expectedOutputs = expectedOutputs;
    for (int i = 0; i < nEpochs; i++) {
        totalloss = 0;  // Reset totalloss at the start of each epoch
        for (int j = 0; j < inputs.size(); j++) {
            feedForward(j, true);
            backPropagation(j);
        }
        totalloss = totalloss / inputs.size();
        std::cout << "Epoch: " << i << " Loss: " << totalloss << std::endl;
    }
    std::cout << "Training complete." << std::endl << std::endl;
}

// Evaluates the neural network on a set of input vectors
// Parameters:
// - entry: a vector of input vectors to evaluate
// Returns:
// - a vector of output vectors corresponding to the inputs
std::vector<std::vector<double>> NeuralNetwork::evaluate(std::vector<std::vector<double>> entry) {
    std::vector<std::vector<double>> res = std::vector<std::vector<double>>(entry.size());
    inputs = entry;
    for (int i = 0; i < entry.size(); i++) {
        feedForward(i, false);
        res[i] = layers[nLayers - 1].getPostactivation();
    }
    return res;
}

// Saves the neural network configuration to a file
// Parameters:
// - filename: the name of the file to save to
void NeuralNetwork::save(std::string filename) {
    std::ofstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return;
    }

    file << nLayers << " " << learningRate << " " << nEpochs << " " << lossFunction << std::endl;
    for (int i = 0; i < nLayers; i++) {
        layers[i].saveLayer(file);
    }

    file.close();

    if (!file) {
        std::cerr << "Error: Unable to close file " << filename << std::endl;
    } else {
        std::cout << "File saved successfully to " << filename << std::endl;
    }
}

// Loads the neural network configuration from a file
// Parameters:
// - filename: the name of the file to load from
void NeuralNetwork::load(const std::string &filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return;
    }

    // Read the first line for general parameters
    std::string line;
    std::getline(file, line);
    std::istringstream iss(line);
    iss >> nLayers >> learningRate >> nEpochs >> lossFunction;
    layers = std::vector<Layer>();
    // Read the blocks of layer data separated by #
    while (std::getline(file, line)) {
        if (line == "#") {
            Layer layer;
            layer = loadLayer(file);
            layers.push_back(layer);
        }
    }

    file.close();
}

// Default constructor for NeuralNetwork
NeuralNetwork::NeuralNetwork() {
    this->learningRate = 0.01;
    this->nEpochs = 100;
    this->nLayers = 0;
    this->totalloss = 0.0;
    this->lossFunction = "mse";
}

// Loads a layer from the input file stream
// Parameters:
// - ifstream: the input file stream to read from
// Returns:
// - the loaded Layer object
Layer NeuralNetwork::loadLayer(std::ifstream& ifstream) {
    std::string line;
    std::getline(ifstream, line);
    std::istringstream iss(line);
    int layerSize;
    std::string activation;
    int lastLayerSize;
    iss >> layerSize;
    iss >> activation;
    iss >> lastLayerSize;
    Layer layer(layerSize, activation);
    layer.setLastLayerSize(lastLayerSize);

    // Read biases
    std::getline(ifstream, line);
    std::istringstream iss2(line);
    std::vector<double> biases = std::vector<double>(layerSize);
    for (int i = 0; i < layerSize; ++i) {
        iss2 >> biases[i];
    }
    layer.setBiases(biases);

    // Read weights
    std::getline(ifstream, line);
    std::istringstream iss3(line);
    std::vector<double> weights = std::vector<double>(layerSize * lastLayerSize);
    for (int i = 0; i < layerSize; ++i) {
        for (int j = 0; j < lastLayerSize; ++j) {
            iss3 >> weights[j + i * lastLayerSize];
        }
    }
    layer.setWeights(weights);
    return layer;
}

// Prints a summary of the neural network configuration
void NeuralNetwork::summary() {
    std::cout << "------------------------------------------------" << std::endl;
    std::cout << std::endl;
    std::cout << "------------------------------------------------" << std::endl;

    std::cout << "Neural Network Summary" << std::endl;
    std::cout << "------------------------------------------------" << std::endl;

    std::cout << "Learning Rate: " << learningRate << std::endl;
    std::cout << "Number of Epochs: " << nEpochs << std::endl;
    std::cout << "Loss Function: " << lossFunction << std::endl;
    std::cout << "Number of Layers: " << nLayers << std::endl;
    std::cout << "------------------------------------------------" << std::endl;
    int params = 0;
    for (int i = 0; i < nLayers; i++) {
        params += layers[i].layerSummary(i);
    }
    std::cout << "Total number of parameters: " << params << std::endl;
    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "------------------------------------------------" << std::endl;
}

// Computes and prints the confusion matrix for the network's performance on given inputs
// Parameters:
// - inputs: a vector of input vectors to evaluate
// - expectedOutputs: a vector of expected output values
void NeuralNetwork::confusion(std::vector<std::vector<double>> inputs, std::vector<double> expectedOutputs) {
    int N = layers[nLayers - 1].getLayerSize();
    this->inputs = inputs;
    this->expectedOutputs = expectedOutputs;
    std::vector<std::vector<double>> confusionMatrix(N, std::vector<double>(N, 0));
    std::vector<std::vector<double>> res = evaluate(inputs);
    for (int i = 0; i < inputs.size(); i++) {
        int max = 0;
        std::vector<double> output = res[i];
        for (int j = 0; j < output.size(); j++) {
            if (output[j] > output[max]) {
                max = j;
            }
        }
        confusionMatrix[expectedOutputs[i]][max]++;
    }
    // Compute accuracy
    double accuracy = 0;
    for (int i = 0; i < N; i++) {
        accuracy += confusionMatrix[i][i];
    }
    accuracy = accuracy / inputs.size();

    // Compute precision & recall
    std::vector<double> precision(N, 0);
    std::vector<double> recall(N, 0);
    for (int i = 0; i < N; i++) {
        double tp = confusionMatrix[i][i];
        double fp = 0;
        double fn = 0;
        for (int j = 0; j < N; j++) {
            if (j != i) {
                fp += confusionMatrix[i][j];
                fn += confusionMatrix[j][i];
            }
        }
        precision[i] = tp / (tp + fp);
        recall[i] = tp / (tp + fn);
    }

    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "Accuracy: " << accuracy << std::endl;
    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "Confusion Matrix: " << std::endl;

    // Determine the maximum width needed for each number
    int maxWidth = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int width = std::to_string(static_cast<int>(confusionMatrix[i][j])).length();
            if (width > maxWidth) {
                maxWidth = width;
            }
        }
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << std::setw(maxWidth) << static_cast<int>(confusionMatrix[i][j]) << " ";
        }
        std::cout << "      | prec_" << i << ": " << precision[i] << "      | recall_"<< i << ": " << recall[i] << std::endl;
    }
}

// Default destructor for NeuralNetwork
NeuralNetwork::~NeuralNetwork() = default;
