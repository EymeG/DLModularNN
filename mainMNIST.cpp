#include <iostream>
#include <fstream>
#include <sstream>
#include "NeuralNetwork.h"

// Function to print the index of the maximum value in a vector
// Parameters:
// - v: the vector of doubles to be evaluated
void printVector(const std::vector<double>& v) {
    int max = 0;
    for (int i = 0; i < v.size(); i++) {
        if (v[i] > v[max]) {
            max = i;
        }
    }
    std::cout << max;
}

// Function to print the expected and actual results of the neural network
// Parameters:
// - inputs: a vector of input vectors
// - results: a vector of output vectors from the neural network
// - expectedOutputs: a vector of expected output values
void printResult(const std::vector<std::vector<double>> inputs, const std::vector<std::vector<double>>& results, const std::vector<double>& expectedOutputs) {
    for (int i = 0; i < inputs.size(); i++) {
        std::cout << " Expected: " << expectedOutputs[i];
        std::cout << " Got: ";
        printVector(results[i]);
        std::cout << std::endl;
    }
}

// Function to load MNIST labels from a CSV file
// Returns:
// - a vector of labels
std::vector<double> loadMNISTLabels() {
    std::ifstream file("./mnist_test.csv");
    std::vector<double> res;
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file mnist_test.csv" << std::endl;
        return res;
    }

    std::string line;
    // Skip the first line (header)
    std::getline(file, line);

    // Read each line of the file
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string label;
        std::getline(ss, label, ',');
        try {
            res.push_back(std::stod(label));
        } catch (const std::invalid_argument& e) {
            std::cerr << "Invalid label found in the file: " << label << std::endl;
        }
    }
    return res;
}

// Function to load MNIST data (images) from a CSV file
// Returns:
// - a vector of vectors, each containing the pixel values of an image
std::vector<std::vector<double>> loadMNISTData() {
    std::ifstream file("./mnist_test.csv");
    std::vector<std::vector<double>> res;
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file mnist_test.csv" << std::endl;
        return res;
    }

    std::string line;
    // Skip the first line (header)
    std::getline(file, line);

    // Read each line of the file
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        std::vector<double> image_data;

        // Skip the first column (label)
        if (std::getline(ss, value, ',')) {
            // Just ignore the label
        }

        // Read the pixel values
        while (std::getline(ss, value, ',')) {
            try {
                image_data.push_back(std::stod(value));
            } catch (const std::invalid_argument& e) {
                std::cerr << "Invalid data found in the file: " << value << std::endl;
            }
        }

        if (!image_data.empty()) {
            res.push_back(image_data);
        }
    }
    return res;
}

// Function to summarize the MNIST data, selecting a subset and reducing the size of the images
// Parameters:
// - data: the full MNIST dataset
// - start: the starting index for the subset
// - size: the number of images to include in the subset
// Returns:
// - a vector of summarized images
std::vector<std::vector<double>> MNISTDataSummarize(std::vector<std::vector<double>> data, int start, int size) {
    std::vector<std::vector<double>> res;
    for (int i = start; i < start + size; i++) {
        std::vector<double> image;
        for (int j = 0; j < 28; j += 2) {
            for (int k = 0; k < 28; k += 2) {
                image.push_back(data[i][j * 28 + k] / 255);
            }
        }
        res.push_back(image);
    }
    return res;
}

// Function to summarize MNIST labels, selecting a subset
// Parameters:
// - labels: the full MNIST label set
// - start: the starting index for the subset
// - size: the number of labels to include in the subset
// Returns:
// - a vector of summarized labels
std::vector<double> MNISTLabelSummarize(std::vector<double> labels, int start, int size) {
    std::vector<double> res;
    for (int i = start; i < size + start; i++) {
        res.push_back(labels[i]);
    }
    return res;
}

int main() {
    int trainsize = 8000;
    int testsize = 10000 - trainsize;

    // Load the full set of MNIST labels and data
    std::vector<double> expectedOutputsFull = loadMNISTLabels();
    std::vector<std::vector<double>> inputsMNISTFull = loadMNISTData();

    // Summarize the data for training and testing
    std::vector<std::vector<double>> inputsMNIST = MNISTDataSummarize(inputsMNISTFull, 0, trainsize);
    std::vector<double> expectedOutputsMNIST = MNISTLabelSummarize(expectedOutputsFull, 0, trainsize);

    std::vector<std::vector<double>> inputTestMNIST = MNISTDataSummarize(inputsMNISTFull, trainsize, testsize);
    std::vector<double> expectedTestMNIST = MNISTLabelSummarize(expectedOutputsFull, trainsize, testsize);

    // Create and configure the neural network
    NeuralNetwork nnMNIST(0.2, 5, "cat_crossentropy");
    nnMNIST.addLayer(Layer(196, "input"));
    nnMNIST.addLayer(Layer(50, "tanh"));
    nnMNIST.addLayer(Layer(15, "sigmoid"));
    nnMNIST.addLayer(Layer(10, "softmax"));
    nnMNIST.summary();

    // Train and save the neural network
    nnMNIST.train(inputsMNIST, expectedOutputsMNIST);
    nnMNIST.save("./models/modelMNIST.txt");

    // Evaluate the neural network on the training data
    std::vector<std::vector<double>> resultsMNIST = nnMNIST.evaluate(inputsMNIST);

    std::cout << std::endl << std::endl;
    std::cout << "--------------------------------------------" << std::endl;
    std::cout << "Results on the training set:" << std::endl;
    nnMNIST.confusion(inputsMNIST, expectedOutputsMNIST);

    // Evaluate the neural network on the test data
    std::cout << std::endl;
    std::cout << "--------------------------------------------" << std::endl;
    std::cout << "Results on the test set:" << std::endl;
    nnMNIST.confusion(inputTestMNIST, expectedTestMNIST);

    std::cout << std::endl;
    std::cout << "--------------------------------------------" << std::endl;
    std::cout << std::endl << "New MNIST network loaded:"<< std::endl;
    NeuralNetwork nnMNIST2;
    nnMNIST2.load("./models/modelMNISTxRelu20xSigm20.txt");

    nnMNIST2.summary();

    std::cout << "Results on the test set:" << std::endl;
    nnMNIST2.confusion(inputTestMNIST, expectedTestMNIST);



    return 0;
}
