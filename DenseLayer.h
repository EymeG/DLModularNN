#ifndef DENSELAYER_H
#define DENSELAYER_H

#include <vector>
#include <string>
#include <fstream>

class Layer {
private:
    int layerSize;                       ///< Number of neurons in the layer
    std::string activation;              ///< Activation function used in the layer
    int lastLayerSize{};                   ///< Number of neurons in the previous layer
    std::vector<double> input;           ///< Input values to the layer
    std::vector<double> weights;         ///< Weights of the layer
    std::vector<double> biases;          ///< Biases of the layer
    std::vector<double> outputs;         ///< Outputs of the layer
    std::vector<double> postactivation;  ///< Outputs after applying the activation function

    /**
     * @brief Generates a random vector of specified size with values in the range [-1, 1].
     * @param size The size of the vector to generate.
     * @return A vector of random values.
     */
    static std::vector<double> generateRandomVector(int size);

    /**
     * @brief Initializes the weights of the layer with random values.
     */
    void initWeights();

    /**
     * @brief Initializes the biases of the layer with random values.
     */
    void initBiases();

public:
    /**
     * @brief Default constructor for Layer.
     */
    Layer();

    /**
     * @brief Constructs a Layer with its size and activation function.
     * @param layerSize The number of neurons in the layer.
     * @param activation The activation function to be used in the layer.
     */
    Layer(int layerSize, std::string activation);

    /**
     * @brief Sets the input values for the layer.
     * @param inputs A vector of input values to be set for the layer.
     */
    void setInputs(const std::vector<double> &inputs);

    /**
     * @brief Returns the size of the layer.
     * @return The number of neurons in the layer.
     */
    [[nodiscard]] int getLayerSize() const;

    /**
     * @brief Computes the outputs of the layer.
     * @param isInputLayer Boolean indicating if the layer is an input layer.
     */
    void computeOutputs(bool isInputLayer);

    /**
     * @brief Calculates the derivative of the activation function for the layer.
     * @return A vector containing the derivatives.
     */
    std::vector<double> getLayerDerivative();

    /**
     * @brief Calculates the derivative of the loss function with respect to the outputs.
     * @param expectedOutput The expected output value.
     * @param lossFunction The loss function to be used ("bin_crossentropy", "mse", "cat_crossentropy").
     * @return A vector containing the loss derivatives.
     */
    std::vector<double> getLossDerivative(const double &expectedOutput, const std::string &lossFunction);

    /**
     * @brief Applies the activation function to the computed outputs.
     */
    void computeActivation();

    /**
     * @brief Updates the weights and biases of the layer based on the computed deltas and learning rate.
     * @param deltas A vector of deltas to update the weights and biases.
     * @param learningRate The learning rate for the update.
     */
    void updateLayerWeightsAndBiases(const std::vector<double> &deltas, double learningRate);

    /**
     * @brief Returns the outputs of the layer.
     * @return A vector containing the outputs.
     */
    [[nodiscard]] std::vector<double> getOutputs() const;

    /**
     * @brief Returns the weights of the layer.
     * @return A vector containing the weights.
     */
    [[nodiscard]] std::vector<double> getWeights() const;

    /**
     * @brief Returns the post-activation outputs of the layer.
     * @return A vector containing the post-activation outputs.
     */
    [[nodiscard]] std::vector<double> getPostactivation() const;

    /**
     * @brief Sets the biases of the layer.
     * @param b A vector of biases to be set for the layer.
     */
    void setBiases(const std::vector<double> &b);

    /**
     * @brief Sets the weights of the layer.
     * @param w A vector of weights to be set for the layer.
     */
    void setWeights(const std::vector<double> &w);

    /**
     * @brief Sets the outputs of the layer.
     * @param o A vector of outputs to be set for the layer.
     */
    void setOutputs(const std::vector<double> &o);

    /**
     * @brief Calculates the deltas for the layer based on the deltas from the next layer.
     * @param lastdeltas A vector of deltas from the next layer.
     * @param nextLayer The next layer object.
     * @return A vector containing the calculated deltas.
     */
    std::vector<double> getDeltas(const std::vector<double> &lastdeltas, Layer &nextLayer);

    /**
     * @brief Computes the loss of the layer.
     * @param expectedOutput The expected output value.
     * @param lossFunction The loss function to be used ("bin_crossentropy", "mse", "cat_crossentropy").
     * @return The calculated loss value.
     */
    double getLoss(double expectedOutput, const std::string &lossFunction);

    /**
     * @brief Sets the size of the last layer.
     * @param size The size of the last layer.
     */
    void setLastLayerSize(int size);

    /**
     * @brief Initializes the layer by initializing its weights and biases.
     */
    void initLayer();

    /**
     * @brief Saves the layer configuration to a file.
     * @param file An ofstream object representing the file to save to.
     */
    void saveLayer(std::ofstream &file);

    /**
     * @brief Prints a summary of the layer configuration and returns the number of parameters.
     * @return The number of parameters in the layer.
     */
    [[nodiscard]] int layerSummary(int index) const;


};

#endif // DENSELAYER_H
