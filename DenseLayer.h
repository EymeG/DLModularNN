//
// Created by eymeg on 04/06/2024.
//

#ifndef TEST_DENSELAYER_H
#define TEST_DENSELAYER_H


#include <vector>
#include <string>

class Layer {
public:
    Layer(int layerSize, const std::string& activation);
    Layer();
    ~Layer() = default;
    void setOutputs(const std::vector<double>& o);
    void setInputs(const std::vector<double>& inputs);
    void initLayer();
    void setBiases(const std::vector<double>& b);
    void setWeights(const std::vector<double>& w);
    void setLastLayerSize(int i);
    [[nodiscard]] int getLayerSize() const;
    [[nodiscard]] std::vector<double> getPostactivation() const;
    [[nodiscard]] std::vector<double> getWeights() const;
    [[nodiscard]] std::vector<double> getOutputs() const;

public:
    void computeOutputs(bool isInputLayer);
    void computeActivation();
    double getLoss(double expectedOutput, const std::string &lossFunction);
    std::vector<double> getLayerDerivative();
    std::vector<double> getLossDerivative(const double& expectedOutput, const std::string &lossFunction);
    void updateLayerWeightsAndBiases(const std::vector<double>& deltas, double learningRate);

public:
    int layerSummary(int i);
    void saveLayer(std::ofstream &file);


private:
    int layerSize;
    int lastLayerSize{};
    std::vector<double> input;
    std::vector<double> weights;
    std::vector<double> biases;
    std::vector<double> outputs;
    std::vector<double> postactivation;
    std::string activation;


private :
    void initBiases();
    void initWeights();
    std::vector<double> getDeltas(const std::vector<double>& lastdeltas, Layer& nextLayer);

private:
    static std::vector<double> generateRandomVector(int size);


};


#endif //TEST_DENSELAYER_H
