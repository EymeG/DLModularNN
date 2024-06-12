cmake_minimum_required(VERSION 3.28)
project(DLModularNN)

set(CMAKE_CXX_STANDARD 17)

add_executable(DLModularNN mainMNIST.cpp
        DenseLayer.cpp
        DenseLayer.h
        NeuralNetwork.h
        NeuralNetwork.cpp
)