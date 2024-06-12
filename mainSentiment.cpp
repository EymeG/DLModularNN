#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <iterator>
#include <cctype>
#include "NeuralNetwork.h"

// Utility function to convert a string to lowercase
std::string toLower(const std::string& str) {
    std::string lowerStr;
    std::transform(str.begin(), str.end(), std::back_inserter(lowerStr), [](unsigned char c){ return std::tolower(c); });
    return lowerStr;
}

// Utility function to split a string into words
std::vector<std::string> split(const std::string& str) {
    std::istringstream iss(str);
    std::vector<std::string> words;
    std::string word;
    while (iss >> word) {
        // Remove punctuation
        word.erase(std::remove_if(word.begin(), word.end(), ::ispunct), word.end());
        words.push_back(toLower(word));
    }
    return words;
}

/// Load sentences and labels from the dataset
std::pair<std::vector<std::string>, std::vector<int>> loadSentencesAndLabels(const std::string& filename, size_t totalToExtract) {
    std::vector<std::string> sentences;
    std::vector<int> labels;
    std::unordered_map<int, size_t> labelCounts; // Count of samples for each label
    std::unordered_map<int, size_t> extractedCounts; // Count of samples extracted for each label

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Could not open the file: " << filename << std::endl;
        return {sentences, labels};
    }

    std::string line;

    // Skip the first line
    std::getline(file, line);
    int total_words = 0;
    while(std::getline(file,line)){
        std::stringstream ss(line);
        std::string label;
        std::string sentence;
        std::string index;
        std::getline(ss, index, ',');
        std::getline(ss, sentence, ',');
        std::getline(ss, label, ',');

        int labelInt = std::stoi(label);
        if (labelCounts.find(labelInt) == labelCounts.end()) {
            labelCounts[labelInt] = 0;
            extractedCounts[labelInt] = 0;
        }
        if (total_words > totalToExtract*6){
            break;
        }
        if (extractedCounts[labelInt] >= totalToExtract) {
            continue;
        }
        sentences.push_back(sentence);
        labels.push_back(labelInt);
        labelCounts[labelInt]++;
        extractedCounts[labelInt]++;
        total_words++;
    }
    file.close();
    return {sentences, labels};
}

// Build vocabulary from the most frequent words
std::unordered_map<std::string, int> buildVocabulary(const std::vector<std::string>& sentences, size_t vocabSize = 200) {
    std::unordered_map<std::string, int> wordFrequency;
    std::unordered_map<std::string, int> commonWords{};
    // Count word frequencies
    for (const auto& sentence : sentences) {
        std::vector<std::string> words = split(sentence);
        for (const auto& word : words) {
            if (commonWords.find(word) == commonWords.end()) {
                wordFrequency[word]++;
            }
        }
    }

    // Create a vector of pairs and sort by frequency
    std::vector<std::pair<std::string, int>> sortedWords(wordFrequency.begin(), wordFrequency.end());
    std::sort(sortedWords.begin(), sortedWords.end(), [](const auto& a, const auto& b) {
        return b.second < a.second; // Sort in descending order of frequency
    });

    // Select the top `vocabSize` words
    std::unordered_map<std::string, int> vocabulary;
    for (size_t i = 0; i < std::min(vocabSize, sortedWords.size()); ++i) {
        vocabulary[sortedWords[i].first] = i + 1; // Key starts from 1
    }

    return vocabulary;
}

// Convert sentences to vectors using the vocabulary and pad to the maximum length
std::vector<std::vector<double>> sentencesToVectors(const std::vector<std::string>& sentences, const std::unordered_map<std::string, int>& vocabulary) {
    std::vector<std::vector<double>> vectors;
    size_t maxLength = 0;

    // Convert each sentence to a vector of numbers
    for (const auto& sentence : sentences) {
        std::vector<std::string> words = split(sentence);
        std::vector<double> vector;
        for (const auto& word : words) {
            if (vocabulary.find(word) != vocabulary.end()) {
                vector.push_back((double) vocabulary.at(word)/(double) vocabulary.size());
            }
        }
        maxLength = std::max(maxLength, vector.size());
        vectors.push_back(vector);
    }

    // Pad vectors to the maximum length
    for (auto& vector : vectors) {
        while (vector.size() < maxLength) {
            vector.push_back(0.0); // Assuming 0 is used for padding
        }
    }

    return vectors;
}

void printVector(const std::vector<double>& v) {
    int max = 0;
    for (int i = 0; i < v.size(); i++) {
        if (v[i] > v[max]) {
            max = i;
        }
    }
    //Six categories: sadness (0), joy (1), love (2), anger (3), fear (4), and surprise (5)
    if (max == 0) {
        std::cout << "sadness (0)" << std::endl;
    } else if (max == 1) {
        std::cout << "joy (1)";
    } else if (max == 2) {
        std::cout << "love (2)";
    } else if (max == 3) {
        std::cout << "anger (3)";
    } else if (max == 4) {
        std::cout << "fear (4)";
    } else if (max == 5) {
        std::cout << "surprise (5)";
    }
}

void printResult(const std::vector<std::vector<double>> inputs, const std::vector<std::vector<double>>& results, const std::vector<double>& expectedOutputs) {
    for (int i = 0; i < inputs.size(); i++) {
        std::cout << " Expected: " << expectedOutputs[i];
        std::cout << " Got: ";
        printVector(results[i]);
        std::cout << std::endl;
    }
}


int main() {
    std::string filename = "./data/sentiments.csv";
    auto [sentences, labels] = loadSentencesAndLabels(filename, 4000);

    // Print the first 5 sentences and labels to verify
    for (size_t i = 0; i < std::min(sentences.size(), size_t(5)); ++i) {
        std::cout << "Sentence: " << sentences[i] << ", Label: " << labels[i] << std::endl;
    }

    // Build vocabulary from the sentences
    std::unordered_map<std::string, int> vocabulary = buildVocabulary(sentences);

    // Print the vocabulary size
    std::cout << "Vocabulary size: " << vocabulary.size() << std::endl;

    // Convert sentences to vectors using the vocabulary
    std::vector<std::vector<double>> vectors = sentencesToVectors(sentences, vocabulary);

    // Print the first 5 vectors to verify
    for (size_t i = 0; i < std::min(vectors.size(), size_t(5)); ++i) {
        std::cout << "Vector: ";
        for (double num : vectors[i]) {
            std::cout << num << " ";
        }
        std::cout << std::endl;
    }

    // Further processing for neural network can be done here
    // For example, splitting the data into training and testing sets, normalizing the data, etc.

    // select the first 8000 sentences for training;
    std::vector<std::vector<double>> trainingInputs(vectors.begin(), vectors.begin() + 8000);
    std::vector<double> trainingOutputs(labels.begin(), labels.begin() + 8000);

    //print the first 5 training inputs and outputs
    for (size_t i = 0; i < std::min(trainingInputs.size(), size_t(5)); ++i) {
        std::cout << "Training Input: ";
        for (double num : trainingInputs[i]) {
            std::cout << num << " ";
        }
        std::cout << "Training Output: " << trainingOutputs[i] << std::endl;
    }

    // select the last 2000 sentences for testing;
    std::vector<std::vector<double>> testingInputs(vectors.begin() + 8000, vectors.end());
    std::vector<double> testingOutputs(labels.begin() + 8000, labels.end());

    NeuralNetwork nn(0.1, 40, "cat_crossentropy");
    nn.addLayer(Layer(trainingInputs[0].size(), "input"));
    nn.addLayer(Layer(20, "sigmoid"));
    nn.addLayer(Layer(20, "sigmoid"));
    nn.addLayer(Layer(20, "sigmoid"));
    nn.addLayer(Layer(6, "softmax"));

    nn.train(trainingInputs, trainingOutputs);
    std::vector<std::vector<double>> results = nn.evaluate(testingInputs);
    nn.summary();
    nn.confusion(testingInputs, testingOutputs);
    nn.save("./models/modelSentiment.txt");




    return 0;
}
