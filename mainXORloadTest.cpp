//
// Created by eymeg on 07/06/2024.
//


#include <iostream>
#include <vector>
#include "DenseLayer.h"
#include "NeuralNetwork.h"

// Function to print the index of the maximum value in a vector
// Parameters:
// - v: the vector of doubles to be evaluated
void printVector(const std::vector<double>& v) {
    for (int i = 0; i < v.size(); i++) {
        std::cout << v[i] << " ";
    }
}

// Function to print the expected and actual results of the neural network
// Parameters:
// - inputs: a vector of input vectors
// - results: a vector of output vectors from the neural network
// - expectedOutputs: a vector of expected output values
void printResult(const std::vector<std::vector<double>> inputs, const std::vector<std::vector<double>>& results, const std::vector<double>& expectedOutputs) {
    for (int i = 0; i < inputs.size(); i++) {
        std::cout << " Input: ";
        printVector(inputs[i]);
        std::cout << " Expected: " << expectedOutputs[i];
        std::cout << " Got: ";
        printVector(results[i]);
        std::cout << std::endl;
    }
}

int main(){

    std::vector<std::vector<double>> inputs = {
            {1, 1},
            {1, 0},
            {0, 1},
            {0, 0}
    };
    std::vector<double> expectedOutputs = {0, 1, 1, 0};


    NeuralNetwork ni = NeuralNetwork();
    ni.load("./models/modelBinaryClassificationXOR.txt");
    ni.summary();
    std::vector<std::vector<double>> results = ni.evaluate(inputs);
    std::cout << "Results after loading the model:" << std::endl;
    printResult(inputs, results, expectedOutputs);


}