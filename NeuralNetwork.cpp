#include "NeuralNetwork.h"
#include "DenseLayer.h"
#include <stdexcept>
#include <iomanip>
#include <sstream>
#include <utility>
#include <algorithm>
#include <chrono>

// Constructor to initialize the neural network
NeuralNetwork::NeuralNetwork(double lr, int nEpochs, std::string lossFunction)
        : learningRate(lr), nEpochs(nEpochs), nLayers(0), totalLoss(0.0), lossFunction(std::move(lossFunction)) {}

// Default constructor
NeuralNetwork::NeuralNetwork()
        : NeuralNetwork(0.01, 100, "mse") {}

// Adds a layer to the neural network
void NeuralNetwork::addLayer(const Layer &layer) {
    layers.push_back(layer);
    nLayers++;
    int prevLayerSize = (nLayers > 1) ? layers[nLayers - 2].getLayerSize() : 0;
    layers.back().setLastLayerSize(prevLayerSize);
    if (nLayers > 1) {
        layers.back().initLayer();
    }
}

// Performs feedforward operation on the network
void NeuralNetwork::feedForward(int inputIndex, bool isTraining) {
    std::vector<double> &input = inputs[inputIndex];
    if (input.size() != layers[0].getLayerSize()) {
        throw std::invalid_argument("Input size does not match input layer size");
    }
    layers[0].setInputs(input);
    layers[0].computeOutputs(true);
    layers[0].computeActivation();
    for (int i = 1; i < nLayers; ++i) {
        layers[i].setInputs(layers[i - 1].getPostactivation());
        layers[i].computeOutputs(false);
        layers[i].computeActivation();
    }
    if (isTraining) {
        totalLoss += layers[nLayers - 1].getLoss(expectedOutputs[inputIndex], lossFunction);
    }
}

// Performs backpropagation to update weights and biases
void NeuralNetwork::backPropagation(int inputIndex) {
    std::vector<std::vector<double>> deltas(nLayers);
    std::vector<double> thisLayerDeltas;
    std::vector<double> thisLayerDerivative = layers[nLayers - 1].getLayerDerivative();
    std::vector<double> thisLossDerivative = layers[nLayers - 1].getLossDerivative(expectedOutputs[inputIndex], lossFunction);

    // Compute delta for the output layer
    thisLayerDeltas.reserve(thisLayerDerivative.size());
    for (size_t i = 0; i < thisLayerDerivative.size(); ++i) {
        thisLayerDeltas.push_back(thisLayerDerivative[i] * thisLossDerivative[i]);
    }
    deltas[nLayers - 1] = thisLayerDeltas;

    // Compute delta for hidden layers
    for (int i = nLayers - 2; i >= 0; --i) {
        thisLayerDeltas.clear();
        const std::vector<double> &nextLayerWeights = layers[i + 1].getWeights();
        for (int j = 0; j < layers[i].getLayerSize(); ++j) {
            double delta = 0.0;
            for (int k = 0; k < layers[i + 1].getLayerSize(); ++k) {
                delta += nextLayerWeights[k * layers[i].getLayerSize() + j] * deltas[i + 1][k];
            }
            delta *= layers[i].getLayerDerivative()[j];
            thisLayerDeltas.push_back(delta);
        }
        deltas[i] = thisLayerDeltas;
    }

    // add multithreading to update bias



    // Update weights and biases for all layers
    for (int i = 1; i < nLayers; ++i) {
        layers[i].updateLayerWeightsAndBiases(deltas[i], learningRate);
    }
}

// Trains the neural network on the provided inputs and expected outputs
void NeuralNetwork::train(const std::vector<std::vector<double>> &inputs, const std::vector<double> &expectedOutputs) {
    std::cout << "Training begin:" << std::endl;
    this->inputs = inputs;
    this->expectedOutputs = expectedOutputs;

    auto startTotal = std::chrono::high_resolution_clock::now(); // Start total timer

    for (int i = 0; i < nEpochs; ++i) {
        auto startEpoch = std::chrono::high_resolution_clock::now(); // Start epoch timer

        totalLoss = 0.0;
        for (size_t j = 0; j < inputs.size(); ++j) {
            feedForward(j, true);
            backPropagation(j);
        }
        totalLoss /= inputs.size();

        auto endEpoch = std::chrono::high_resolution_clock::now(); // End epoch timer
        auto durationEpoch = std::chrono::duration_cast<std::chrono::seconds>(endEpoch - startEpoch).count();

        std::cout << "Epoch: " << i << " Loss: " << totalLoss
                  << " Time: " << durationEpoch << "s" << std::endl;
    }

    auto endTotal = std::chrono::high_resolution_clock::now(); // End total timer
    auto durationTotal = std::chrono::duration_cast<std::chrono::seconds>(endTotal - startTotal).count();

    std::cout << "Training complete. Total time: " << durationTotal << "s" << std::endl << std::endl;
}

// Evaluates the neural network on a set of input vectors
std::vector<std::vector<double>> NeuralNetwork::evaluate(const std::vector<std::vector<double>> &entry) {
    std::vector<std::vector<double>> results(entry.size());
    inputs = entry;
    for (size_t i = 0; i < entry.size(); ++i) {
        feedForward(i, false);
        results[i] = layers[nLayers - 1].getPostactivation();
    }
    return results;
}

// Saves the neural network configuration to a file
void NeuralNetwork::save(const std::string &filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return;
    }
    file << nLayers << " " << learningRate << " " << nEpochs << " " << lossFunction << std::endl;
    for (Layer layer : layers) {
        layer.saveLayer(file);
    }
    file.close();
    std::cout << "Neural network saved to " << filename << std::endl;
}

// Loads the neural network configuration from a file
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

// Loads a single layer from an input file stream
Layer NeuralNetwork::loadLayer(std::ifstream& ifstream) {
    std::string line;
    std::getline(ifstream, line);
    std::istringstream iss(line);
    int layerSize;
    std::string activation;
    int lastLayerSize;
    iss >> layerSize >> activation >> lastLayerSize;
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
void NeuralNetwork::summary() const {
    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "Network Configuration:" << std::endl;
    std::cout << "Learning Rate: " << learningRate << std::endl;
    std::cout << "Number of Epochs: " << nEpochs << std::endl;
    std::cout << "Loss Function: " << lossFunction << std::endl;
    std::cout << "Number of Layers: " << nLayers << std::endl;
    int totalParams = 0;
    for (int i = 0; i < nLayers; ++i){
        Layer layer = layers[i];
        totalParams += layer.layerSummary(i);
    }
    std::cout << "Total number of parameters: " << totalParams << std::endl;
    std::cout << "------------------------------------------------" << std::endl;
}

// Computes and prints the confusion matrix for the network's performance on given inputs
void NeuralNetwork::confusion(const std::vector<std::vector<double>> &inputs, const std::vector<double> &expectedOutputs) {
    int N = layers[nLayers - 1].getLayerSize();
    this->inputs = inputs;
    this->expectedOutputs = expectedOutputs;
    std::vector<std::vector<double>> confusionMatrix(N, std::vector<double>(N, 0));
    std::vector<std::vector<double>> results = evaluate(inputs);
    for (size_t i = 0; i < inputs.size(); ++i) {
        int maxIndex = std::distance(results[i].begin(), std::max_element(results[i].begin(), results[i].end()));
        confusionMatrix[static_cast<int>(expectedOutputs[i])][maxIndex]++;
    }

    // Compute accuracy
    double accuracy = 0;
    for (int i = 0; i < N; ++i) {
        accuracy += confusionMatrix[i][i];
    }
    accuracy /= inputs.size();

    // Compute precision & recall
    std::vector<double> precision(N, 0);
    std::vector<double> recall(N, 0);
    for (int i = 0; i < N; ++i) {
        double tp = confusionMatrix[i][i];
        double fp = 0;
        double fn = 0;
        for (int j = 0; j < N; ++j) {
            if (j != i) {
                fp += confusionMatrix[i][j];
                fn += confusionMatrix[j][i];
            }
        }
        precision[i] = tp / (tp + fp);
        recall[i] = tp / (tp + fn);
    }

    // Print results
    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "Accuracy: " << accuracy << std::endl;
    std::cout << "Confusion Matrix: " << std::endl;

    // Determine the maximum width needed for each number
    int maxWidth = 0;
    for (const auto &row : confusionMatrix) {
        for (const auto &elem : row) {
            int width = std::to_string(static_cast<int>(elem)).length();
            if (width > maxWidth) {
                maxWidth = width;
            }
        }
    }

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << std::setw(maxWidth) << static_cast<int>(confusionMatrix[i][j]) << " ";
        }
        // force to print precesion with 6 decimals:
        std::cout << " | prec_" << i << ": " << std::fixed << std::setprecision(6) << precision[i] << " | recall_" << i << ": " << recall[i] << std::endl;
    }
}

// Destructor
NeuralNetwork::~NeuralNetwork() = default;