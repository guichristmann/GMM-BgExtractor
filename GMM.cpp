/* 
 * This implementation is based on the paper "Adaptive background mixture models for real-time tracking" by C. Stauffer and W.E.L. Grimson.
 * https://ieeexplore.ieee.org/abstract/document/784637
*/

#include <cmath>
#include <iostream>
#include <vector>
#include <random>
#include <numeric>
#include <algorithm>
#include "GMM.hpp"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

double Gaussian::getProbability(double dist){
    return (1.0 / sqrt(2 * M_PI * variance)) *
           exp(-dist / (2 * variance));
}

Gaussian::Gaussian(){}

/* Utility functions */

/* Retrieve N random samples from given vector */
vector<Vec3b> randomSample(vector<Vec3b> data, int N){
    // Instantiate random engine
    random_device rd;
    mt19937 eng(rd());
    uniform_int_distribution<> generator(0, data.size()-1);

    // Create vector to return the random samples
    vector<Vec3b> sampled;
    // Keeps track of which indices have already been sampled to avoid resampling
    vector<int> sampledIndices;
    // Generate N random numbers
    int index;
    bool alreadySampled = false;
    while (sampled.size() < N){
        // Generate a random index
        index = generator(eng);
        // Check if this index has already been sampled 
        alreadySampled = false;
        for (auto itr = sampledIndices.begin(); itr != sampledIndices.end(); ++itr){
            if (index == *itr){
                alreadySampled = true;
                break;
            }
        }

        if (alreadySampled)
            continue; // Skip index
        else{
            // Retrieve data at index and push into sampled
            sampled.push_back(data.at(index));
            sampledIndices.push_back(index);
        }
    }

    return sampled;
}

/* Calculate variance over vector of doubles */
double calcVariance(vector<Vec3b> data, double mean){
    double sqDiffs = 0.0;
    for(int i = 0; i < data.size(); ++i) {
        double sqDist = (data.at(i)[0] - mean) + (data.at(i)[1] - mean) + (data.at(i)[2] - mean);
        sqDiffs += sqDist; 
    }

    return sqDiffs / data.size();
}

// Compare function used to sort the array of Gaussian distributions. Will sort in descending order
// based on weight/variance
int distCmp(const void *_g1, const void *_g2){
    Gaussian *g1 = (Gaussian*) _g1;
    Gaussian *g2 = (Gaussian*) _g2;

    if (g1->weightStdRatio < g2->weightStdRatio){
        return 1;
    } else if (g1->weightStdRatio == g2->weightStdRatio){
        return 0;
    } else {
        return -1;
    }
}

GMM::GMM(int nComponents){
    K = nComponents;
}

void GMM::init(vector<Vec3b> initialSamples){
    // Get K samples from initialSamples to initialize as the mean
    vector<Vec3b> means = randomSample(initialSamples, K);

    // Create K Gaussians
    for(int i = 0; i < K; i++){
        // Parameters for the new Gaussian
        Gaussian g;
        g.meanB = (double) means.at(i)[0];
        g.meanG = (double) means.at(i)[1];
        g.meanR = (double) means.at(i)[2];

        g.variance = calcVariance(initialSamples, (g.meanB+g.meanG+g.meanR)/3.0);
        if (g.variance < minVariance){
            g.variance = minVariance;
        }
        g.weight = 1.0/K;
        g.weightStdRatio = g.weight / sqrt(g.variance);
        gDists.push_back(g);
    }
    // Reorder Gaussians by weight/std
    qsort(gDists.data(), K, sizeof(Gaussian), distCmp);
}

// Update GMM with new value. Returns true if the observed pixel is background and false
// if it's foreground
int GMM::update(Vec3b pixelVal, double bgThresh){
    // Will return wether observed pixel is background or foreground
    // 1 means background, 0 means foreground
    int background = 0;

    // Find index of gDists until which we consider background distributions. Assuming the
    // order is in descending order
    int bgGaussianI = 0;
    double sum = 0.0;
    for(int i = 0; i < K; i++){
        if(sum < bgThresh){
            bgGaussianI++;
            sum += gDists.at(i).weight;
        } else{
            break;
        }
    }

    // Flag used when already found a match for the observed value
    bool foundMatch = false;
    double weightSum = 0.0; // Keeps track of total weight, to renormalize after updating
    // Iterate over gaussians, updating and finding match for the observed value
    for (int i = 0; i < K; i++){
        // Parameters of current distribution
        double weight = gDists.at(i).weight;
        double meanB = gDists.at(i).meanB;
        double meanG = gDists.at(i).meanG;
        double meanR = gDists.at(i).meanR;
        double variance = gDists.at(i).variance;

        // Note: not taking the square root of this for optimization. In the comparin below the sqrt of variance is not taken, so the operations become equivalent
        double dist = pow(meanB - pixelVal[0], 2) + pow(meanG - pixelVal[1], 2) + pow(meanR - pixelVal[2], 2);

        if (dist < 7.5 * variance && i < bgGaussianI){
            background = 1;
        }

        // Already found a match so we only update the weight for this distribution
        if (foundMatch){
            gDists.at(i).weight = (1-lr) * weight;
            if (gDists.at(i).weight < 1e-8){
                cout << "Capping weight.\n";
                gDists.at(i).weight = 1e-8;
            }
        }
        // A match occurs when the distance between the value is less than 2.5*std according to the paper.
        // Not taking the square root of variance (std) because dist is already squared
        else if (dist < 2.5 * variance){ 
            foundMatch = true;
            // If matched gaussian is one of the background gaussians
            if (i < bgGaussianI){
                // Set return flag to true
                background = 1;
            }

            // Update the parameters of the matched gaussian
            gDists.at(i).weight = (1-lr) * weight + lr; 
            double p = lr * gDists.at(i).getProbability(dist);
            gDists.at(i).meanB = (1-p) * meanB + p * pixelVal[0];
            gDists.at(i).meanG = (1-p) * meanG + p * pixelVal[1];
            gDists.at(i).meanR = (1-p) * meanR + p * pixelVal[2];
            gDists.at(i).variance = (1-p) * variance + p * (dist - variance);

            if (gDists.at(i).variance < minVariance){
                gDists.at(i).variance = minVariance;
            }else if(gDists.at(i).variance > 5*highVariance){
                gDists.at(i).variance = 5*highVariance;
            }
        }
        weightSum += gDists.at(i).weight;
    }

    // If there was no matching distribution, replace the last distribution, using current value as mean and a high variance
    if(!foundMatch){
        Gaussian g;
        g.meanB = pixelVal[0];
        g.meanG = pixelVal[1];
        g.meanR = pixelVal[2];
        g.variance = highVariance; 
        g.weight = gDists.at(K-1).weight;
        g.weightStdRatio = g.weight / sqrt(g.variance);

        // Replace last distribution in vector with new one
        gDists.at(K-1) = g;
    }

    // Renormalize weights to sum to 1.0
    for(int i = 0; i < K; ++i){
        gDists.at(i).weight = gDists.at(i).weight / weightSum;
        gDists.at(i).weightStdRatio = gDists.at(i).weight / sqrt(gDists.at(i).variance);
    }

    // Reorder Gaussians by weight/std
    qsort(gDists.data(), K, sizeof(Gaussian), distCmp);

    return background;
}
