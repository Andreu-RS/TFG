#pragma once

#include <string>
#include <iostream>

// Definitions
#define MIN(a,b) (((a)<=(b))?(a):(b))
#define MAX(a,b) (((a)>=(b))?(a):(b))

// Auxiliar Structures
class Penalties {
    public:
        int match;
        int mismatch;
        int insertion;
        int deletion;

        Penalties(){
            this->match = 0; 
            this->mismatch = 5;
            this->insertion = 2;
            this->deletion = 3;
        };
};

void nw_align(
    const Penalties penalties,
    const std::string query,
    const uint16_t queryLength,
    const std::string target,
    const uint16_t targetLength,
    int** matrix);

void nw_out(
    int nAligments,
    std::string query,
    uint16_t queryLength,
    std::string target,
    uint16_t targetLength,
    int** matrix);