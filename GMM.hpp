#include <vector>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

vector<Vec3b> randomSample(vector<Vec3b>, int);
double calcVariance(vector<Vec3b>, double);
double gaussian(double, double, double);

class Gaussian{
    public:
        double meanB;
        double meanG;
        double meanR;
        double variance;
        double weight;
        double weightStdRatio; // Used to sort the distributions

        Gaussian();
        double getProbability(double dist);
};

class GMM{
    public:
        // Number of Gaussian Components
        int K;

        // Vector of gaussian distributions
        vector<Gaussian> gDists;

        // Learning rate/Update factor
        double lr = 0.001;

        // When there's no match for any of the current Gaussian distributions the least probable
        // one is replaced based on the current observation. The paper states that we use the
        // current value as the mean and "an initially high variance" defined below. A good value
        // might be very dependent on the domain of the problem we're modelling.
        // The variance is also limited so we don't get crazy big numbers, and hard to match distributions.
        double highVariance = 36.0;
        double minVariance = 8.0;

        GMM(int nComponents);
        void init(vector<Vec3b> initialSamples);
        int update(Vec3b value, double bgThresh);
};
