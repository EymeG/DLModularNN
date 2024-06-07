//
// Created by eymeg on 04/06/2024.
//

#ifndef TEST_NEURALNETWORK_H
#define TEST_NEURALNETWORK_H


#include <vector>
#include <string>
#include "DenseLayer.h"

class NeuralNetwork {
public:
    ~NeuralNetwork();
    NeuralNetwork(double lr, int nEpochs, std::string lossFunction);
    NeuralNetwork();
    void addLayer(const Layer& l);
    void train(const std::vector<std::vector<double>>& inputs, const std::vector<double>& expectedOutputs);
    std::vector<std::vector<double>> evaluate(std::vector<std::vector<double>> entry);
    void summary();
    void save(std::string filename);
    void load(const std::string &filename);
    void confusion(std::vector<std::vector<double>> inputs, std::vector<double> expectedOutputs);

private:
    std::string lossFunction = "mse";
    double learningRate;
    int nEpochs;
    int nLayers = 0;
    double totalloss = 0;
    std::vector<Layer> layers;
    std::vector<std::vector<double>> inputs;
    std::vector<double> expectedOutputs;


private:
    void backPropagation(int inputIndex);
    static Layer loadLayer(std::ifstream &ifstream);
    void feedForward(int inputIndex, bool isTraining);

};


#endif //TEST_NEURALNETWORK_H
