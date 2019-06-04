#include "GMM.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <ctime>

using namespace cv;
using namespace std;

#define INITIAL_FRAMES 10 // How many frames are used as initialization data for the GMM
#define LEARNING_RATE 0.003 // Alpha parameter in paper
#define N_GAUSSIANS 7 // How many gaussians created for each pixel
#define BG_T 0.75 // Threshold for removing the background. Proportion of weights to be considered background
#define RESIZE_FACTOR 0.5 // Factor to resize the image

int main(int argc, char* argv[]){
    // Open video stream
    VideoCapture cap(0);

    if (!cap.isOpened()){
        cout << "Failed to open video file!\n";
        return -1;
    }

    // Get first frame outside of loop so we can get some information
    Mat frame;
    cap >> frame;
    resize(frame, frame, Size(), RESIZE_FACTOR, RESIZE_FACTOR, INTER_CUBIC);

    // Retrieving frame information
    unsigned int rows = frame.rows;
    unsigned int cols = frame.cols;
    unsigned int imgSize = rows * cols;

    cout << "Image dimensions: " << rows << "x" << cols << endl;

    // Vector with initialization data for GMM. A vector of pixel data over the initial
    // frames
    vector<vector<Vec3b>> initialData(imgSize);
    Vec3b pixelVal;
    for(int i = 0; i < INITIAL_FRAMES; ++i){
        int pixelCount = 0;
        for(int r = 0; r < rows; ++r){
            for(int c = 0; c < cols; ++c){
                pixelVal = frame.at<Vec3b>(r, c);
                initialData.at(pixelCount).push_back(pixelVal);
                pixelCount++;
            }
        }

        cap >> frame;
        resize(frame, frame, Size(), RESIZE_FACTOR, RESIZE_FACTOR, INTER_CUBIC);
    }

    // Create GMM for each pixel in the image
    cout << "Creating GMM for each pixel" << endl;
    //GMM* pixelGMM = (GMM*) malloc(imgSize*sizeof(GMM));
    vector<GMM> pixelGMM;
    int pixelCount = 0;
    for(int r = 0; r < rows; ++r){
        for(int c = 0; c < cols; ++c){
            GMM gmm(N_GAUSSIANS);
            gmm.init(initialData.at(pixelCount));
            gmm.lr = LEARNING_RATE;
            pixelGMM.push_back(gmm);
            pixelCount++; 
        }
    }

    cout << "Done." << endl;

    // Process subsequent frames in the video performing background subtraction
    VideoWriter video("video.avi", CV_FOURCC('M', 'J', 'P', 'G'), 24, Size(cols, rows));
    while (true){
        // Capture new frame and resize
        cap >> frame;
        if (frame.empty()){
            break;
        }
        resize(frame, frame, Size(), RESIZE_FACTOR, RESIZE_FACTOR, INTER_CUBIC);

        Mat gray;
        Mat resultImg = Mat::zeros(Size(cols, rows), CV_8UC3); // Resulting mat
        Mat bgMask = Mat::zeros(Size(cols, rows), CV_8U); // Mask mat

        unsigned int pixelCount = 0;
        // Retrieve pixel values and update GMM
        for(int r = 0; r < rows; ++r){
            for(int c = 0; c < cols; ++c){
                // Get new observation (pixel value)
                pixelVal = frame.at<Vec3b>(r, c);

                // Update GMM and check if pixel is background or foreground
                int bgResponse = pixelGMM[pixelCount].update(pixelVal, BG_T);
                if(bgResponse == 1) {
                    bgMask.data[pixelCount] = 0;
                }else if (bgResponse == 0){
                    bgMask.data[pixelCount] = 255;
                }
                pixelCount++;
            }
        }

        // Aply mask to original color frame
        bitwise_and(frame, frame, resultImg, bgMask);

        // Show frame
        imshow("Frame", frame);
        imshow("Mask", bgMask);
        imshow("Result", resultImg);

        waitKey(1);
    }

    return 0;
}
