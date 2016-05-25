//Timur Galimov
//Wei Zhang

//#include "opencv2\opencv.hpp"
#include "opencv2\objdetect\objdetect.hpp"
#include "opencv2\core.hpp"
#include "opencv2\face.hpp"
#include "opencv2\highgui.hpp"
#include "opencv2\imgproc.hpp"
#include <cstdlib>
#include <conio.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <string>

using namespace cv;
using namespace cv::face;
using namespace std;

Mat croppedImage;
void showHistogram(vector<Mat> hist)
{
    int bins = 256;             // number of bins
    int nc = 1;
    vector<Mat> canvas(nc);     // images for displaying the histogram
    int hmax[3] = { 0, 0, 0 };      // peak value for each histogram

    for (int i = 0; i < nc; i++)
    {
        for (int j = 0; j < bins - 1; j++)
            hmax[i] = hist[i].at<int>(j) > hmax[i] ? hist[i].at<int>(j) : hmax[i];
    }

    const char* wname[3] = { "blue", "green", "red" };
    Scalar colors[3] = { Scalar(255, 0, 0), Scalar(0, 255, 0), Scalar(0, 0, 255) };

    for (int i = 0; i < nc; i++)
    {
        canvas[i] = Mat::ones(125, bins, CV_8UC3);

        for (int j = 0, rows = canvas[i].rows; j < bins - 1; j++)
        {
            line(
                    canvas[i],
                    Point(j, rows),
                    Point(j, rows - (hist[i].at<int>(j) * rows / hmax[i])),
                    nc == 1 ? Scalar(200, 200, 200) : colors[i],
                    1, 8, 0
            );
        }

        imshow(nc == 1 ? "value" : wname[i], canvas[i]);
    }
}
int main(int agrc, const char** agrv)
{
    //Step 1 Data mining

    //create cascade classifier object for Face detection
    CascadeClassifier face_cascade;
    //use haarcascade_frantalface_alt.xml library
    face_cascade.load("haarcascade_frontalface_alt.xml");

    //set video capture
    VideoCapture captureDevice;
    captureDevice.open(0);
    //CvCapture* capture;
    //capture = cvCaptureFromCAM(-1);

    //create image Files use videocapture
    Mat captureFrame;
    Mat grayscaleFrame;

    //window to present results
    namedWindow("outputCapture", 1);

    cout << "Step 1: Evualuation of Learn Images \n";
    cout << "For each face we collect 15 samples" << endl;
    cout << "Press Space- Key to save face from video frame\n";
    cout << "Press ESC to finish Step 1\n";

    cout << "Now first face!!!" << endl;
    int counter = 0;
    // create a loop Capture to find a faces
    while (counter < 30)
    {
        //Create a new image frame
        captureDevice >> captureFrame;
        //captureFrame = cvQueryFrame( capture );
        // Create Vector to Store The faces
        vector <Rect> faces;
        if (!captureFrame.empty()) {
            // Convert captured image to Grayscale and equalize
            cvtColor(captureFrame, grayscaleFrame, CV_BGR2GRAY);
            //equalizeHist(grayscaleFrame, grayscaleFrame);

            //Fund Face
            face_cascade.detectMultiScale(grayscaleFrame, faces, 1.1, 3, CV_HAAR_FIND_BIGGEST_OBJECT | CV_HAAR_SCALE_IMAGE, Size(20, 20));
        }

        for (int i = 0; i < faces.size(); i++)
        {
            //Draw rectangle on Selected Image
            Point pt1(faces[i].x + faces[i].width, faces[i].y + faces[i].height);
            Point pt2(faces[i].x, faces[i].y);

            rectangle(captureFrame, pt1, pt2, cvScalar(0, 255, 0, 0), 1, 8, 0);
            Rect myROI(pt1, pt2);

            croppedImage = captureFrame(myROI);
        }

        imshow("outputCapture", captureFrame);

        char file_name[256];
        int key = waitKey(10);
        if (key == 32)
        {
            cout << "Save the " << counter % 15 + 1 << "th sample for face " << counter / 15 + 1 << endl;
            sprintf(file_name, "face//test_%d.jpg", counter++);
            imwrite(file_name, croppedImage);
            if (counter == 15) cout << "Now second face!!!" << endl;
        }
        else if (key == 8)
        {
            cout << "Save the " << counter % 15 + 1 << "th sample for face" << counter / 15 + 1 << endl;
            sprintf(file_name, "face//test_%d.jpg", counter++);
            imwrite(file_name, croppedImage);
            if (counter == 15) cout << "Now second face!!!" << endl;

        }
        else if (key == 27) break;
        else continue;
    }

    //�//////////////////// Learning ///////////////////////
    cout << "Step 2, Training of Classifier\n";

    vector <Mat> images;
    vector <int> labels;

    std::cout << "Load train Images: " << endl;

    for (int i = 0; i < 15; i++)
    {
        string name = format("face//test_%d.jpg", i);
        Mat img = imread(name, 0);
        if (img.empty())
        {
            cerr << name << " can't be loaded!" << endl;
        }

        images.push_back(img);
        labels.push_back((int)1);
    }

    for (int i = 15; i < 29; i++)
    {
        string name = format("face//test_%d.jpg", i);
        Mat img = imread(name, 0);
        if (img.empty())
        {
            cerr << name << " can't be loaded!" << endl;
        }
        images.push_back(img);
        labels.push_back((int)2);
    }
    for (int i = 1; i < 11; i++)
    {
        string name = format("otherface//%d.pgm", i);
        Mat img = imread(name, 0);
        if (img.empty())
        {
            cerr << name << " can't be loaded!" << endl;
        }
        images.push_back(img);
        labels.push_back((int)3);
    }

    std::cout << images.size() << endl;
    std::cout << labels.size() << endl;

    //train classifier
    std::cout << "Training LBPHF Classifier..." << endl;
    Ptr<LBPHFaceRecognizer> model = createLBPHFaceRecognizer();
    //Ptr<BasicFaceRecognizer> model = createEigenFaceRecognizer();
    model->train(images, labels);
    // Show some informations about the model, as there's no cool
    // Model data to display as in Eigenfaces/Fisherfaces.
    // Due to efficiency reasons the LBP images are not stored
    // within the model:
    cout << "Model Information:" << endl;
    string model_info = format("\tLBPH(radius=%i, neighbors=%i, grid_x=%i, grid_y=%i, threshold=%.2f)",
                               model->getRadius(),
                               model->getNeighbors(),
                               model->getGridX(),
                               model->getGridY(),
                               model->getThreshold());
    cout << model_info << endl;
    // We could get the histograms for example:
    vector<Mat> histograms = model->getHistograms();
    //showHistogram(histograms);
    // But should I really visualize it? Probably the length is interesting:
    cout << "Size of the histograms: " << histograms[0].total() << endl;
    std::cout << "Done" << endl;

    ////////////////////// Prediction ////////////////////////////
    //person name
    string Pname, Pname1, Pname2;
    cout << "Please give the person name of face one and two: " << endl;
    cin >> Pname1 >> Pname2;

    while (true)
    {
        //Create a new image frame
        captureDevice >> captureFrame;

        // Convert captured image to Grayscale and equalize
        cvtColor(captureFrame, grayscaleFrame, CV_BGR2GRAY);

        // Create Vector to Store The faces
        vector <Rect> faces;

        //Fund Face
        face_cascade.detectMultiScale(grayscaleFrame, faces, 1.1, 3, CV_HAAR_FIND_BIGGEST_OBJECT | CV_HAAR_SCALE_IMAGE, Size(20, 20));

        //Draw rectangle on Selected Image
        int label = -1;
        double confidence = 0.0;
        for (int i = 0; i < faces.size(); i++)
        {
            Point pt1(faces[i].x + faces[i].width, faces[i].y + faces[i].height);
            Point pt2(faces[i].x, faces[i].y);

            rectangle(captureFrame, pt1, pt2, cvScalar(0, 255, 0, 0), 3, 8, 0);
            Rect myROI(pt1, pt2);

            croppedImage = captureFrame(myROI);
            cvtColor(croppedImage, croppedImage, CV_BGR2GRAY);
            model->predict(croppedImage, label, confidence);
            string text = "Detected";
            if (label == 1)
            {
                //string text = format("Person is  = %d", label);
                Pname = Pname1;
            }
            else if (label == 2)
            {
                Pname = Pname2;
            }
            else
            {
                Pname = "Unbekannt";
            }

            cout << pt1, pt2;
            putText(captureFrame, text, Point(15, 15), CV_FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(0, 255, 0), 1.0);
        }
        putText(captureFrame, Pname, Point(30, 90), CV_FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(0, 255, 0), 1.0);
        imshow("outputCapture", captureFrame);

        if (waitKey(10) == 27)
            break;
    }
    return 0;
}
