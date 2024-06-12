#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include "DenseLayer.h"

class NeuralNetwork {
public:
    /**
     * @brief Constructs a NeuralNetwork with the given learning rate, number of epochs, and loss function.
     * @param lr The learning rate for the neural network.
     * @param nEpochs The number of epochs for training.
     * @param lossFunction The loss function to use for training ("mse", "cross_entropy", etc.).
     */
    NeuralNetwork(double lr, int nEpochs, std::string lossFunction);

    /**
     * @brief Default constructor for NeuralNetwork, initializes with default values.
     */
    NeuralNetwork();

    /**
     * @brief Adds a layer to the neural network.
     * @param layer The Layer object to add.
     */
    void addLayer(const Layer &layer);

    /**
     * @brief Performs a feedforward operation through the network.
     * @param inputIndex The index of the input data to use.
     * @param isTraining Flag to indicate if the network is in training mode.
     */
    void feedForward(int inputIndex, bool isTraining);

    /**
     * @brief Performs backpropagation to update weights and biases.
     * @param inputIndex The index of the input data to use.
     */
    void backPropagation(int inputIndex);

    /**
     * @brief Trains the neural network using the provided inputs and expected outputs.
     * @param inputs A vector of input vectors for training.
     * @param expectedOutputs A vector of expected output values corresponding to the inputs.
     */
    void train(const std::vector<std::vector<double>> &inputs, const std::vector<double> &expectedOutputs);

    /**
     * @brief Evaluates the neural network on the given set of input vectors.
     * @param entry A vector of input vectors to evaluate.
     * @return A vector of output vectors produced by the network.
     */
    std::vector<std::vector<double>> evaluate(const std::vector<std::vector<double>> &entry);

    /**
     * @brief Saves the neural network configuration to a file.
     * @param filename The name of the file to save the network configuration.
     */
    void save(const std::string &filename) const;

    /**
     * @brief Loads the neural network configuration from a file.
     * @param filename The name of the file to load the network configuration.
     */
    void load(const std::string &filename);

    /**
     * @brief Prints a summary of the neural network configuration.
     */
    void summary() const;

    /**
     * @brief Computes and prints the confusion matrix for the network's performance.
     * @param inputs A vector of input vectors for evaluation.
     * @param expectedOutputs A vector of expected output values corresponding to the inputs.
     */
    void confusion(const std::vector<std::vector<double>> &inputs, const std::vector<double> &expectedOutputs);

    /**
     * @brief Destructor for the NeuralNetwork class.
     */
    ~NeuralNetwork();

private:
    double learningRate;            ///< The learning rate for the neural network.
    int nEpochs;                    ///< The number of epochs for training.
    int nLayers;                    ///< The number of layers in the neural network.
    double totalLoss;               ///< The total loss accumulated during training.
    std::string lossFunction;       ///< The loss function used for training.
    std::vector<Layer> layers;      ///< The layers of the neural network.
    std::vector<std::vector<double>> inputs; ///< The input data for training or evaluation.
    std::vector<double> expectedOutputs;     ///< The expected output values for training.

    /**
     * @brief Loads a single layer from an input file stream.
     * @param ifstream The input file stream to read from.
     * @return The loaded Layer object.
     */
    static Layer loadLayer(std::ifstream &ifstream) ;
};

#endif // NEURALNETWORK_H