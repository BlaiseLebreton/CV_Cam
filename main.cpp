#include <iostream>
#include <sstream>
#include <time.h>
#include <math.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <opencv2/features2d.hpp>

using namespace cv;
using namespace std;

#define WIDTH 640
#define HEIGHT 480

int cmin = 130, cmax = 1000; // maxi/mini longeur contour
int niter = 3; // Number of erosion/dilatation

int main(int argc, char* argv[]) {

  int x,y,np;
  // Execution time
  double start,stop, dt;

  // Create Background Subtractor objects
  Ptr<BackgroundSubtractor> pBackSub;
  // (history, varThreshold, detectShadows);
  pBackSub = createBackgroundSubtractorMOG2(500, 16, false);
  // pBackSub = createBackgroundSubtractorKNN(500, 16, false);

  // Create capture
  VideoCapture capture("video.mp4");
  capture.set(CAP_PROP_FRAME_WIDTH,  WIDTH);
  capture.set(CAP_PROP_FRAME_HEIGHT, HEIGHT);

  // Create window
  namedWindow("Raw", WINDOW_NORMAL);
  resizeWindow("Raw", WIDTH, HEIGHT);

  if (!capture.isOpened()){
    cerr << "Unable to open file" << endl;
    return 0;
  }

  Mat raw, filtered, decision, display;
  Mat obj = Mat::zeros(Size(WIDTH, HEIGHT), CV_64FC1);
  while (true) {

    // Frame rate
    start = getTickCount();

    capture >> raw;
    if (raw.empty()) {
      break;
    }

    // Creating display image
    cvtColor(raw, display, COLOR_BGR2GRAY);
    cvtColor(display, display, COLOR_GRAY2BGR);

    // Apply blur
    GaussianBlur(raw, filtered, Size(11, 11), 0, 0);

    // Update the background model
    pBackSub->apply(filtered, decision);

    // Erosion / Dilatation
    for (int i=0; i<niter; i++) {
      dilate(decision,decision,getStructuringElement(MORPH_ELLIPSE,Size(5,5)));
    }
    for (int i=0; i<niter; i++) {
      erode( decision,decision,getStructuringElement(MORPH_ELLIPSE,Size(5,5)));
    }

    // Contour
    vector<vector<Point>> contours;
    vector<Point> Centers;
    np = 0;
    findContours(decision,contours,RETR_EXTERNAL,CHAIN_APPROX_NONE);

    // Elimination des contours trop long/court
    vector<vector<Point>>::iterator itc = contours.begin();

    while (itc != contours.end()) {

      if (itc->size() < cmin || itc->size() > cmax) {
        itc = contours.erase(itc);
      }
      else {
        // Centre de gravite
        Moments mom = moments(*itc);
        x = mom.m10/mom.m00;
        y = mom.m01/mom.m00;
        circle(display, Point(x,y), 2, Scalar(0,0,255), 2);
        Centers.push_back(Point(x,y));
        np++;
        ++itc;
      }
    }
    // drawContours(display, contours, -1, Scalar(0,0,255), 2);

    // Applying gaussian over centers
    for (int p = 0; p < np; p++) {
      // Centers.at(p).x / Centers.at(p).y



    }




    // Trackbars
    createTrackbar("Niter", "Raw", &niter, 10);
    createTrackbar("Min Size", "Raw", &cmin, 1000);
    createTrackbar("Max Size", "Raw", &cmax, 1000);


    // Frame rate
    stop = getTickCount();
    dt = ((stop - start)/ getTickFrequency());
    putText(display, to_string((int)(1/dt)) + "Hz", Point(30,30), FONT_HERSHEY_DUPLEX, 1.0, Scalar(255,255,255), 2);

    // Display result
    imshow("Raw", display);

    int keyboard = waitKey(30);
    if (keyboard == 'q' || keyboard == 27) {
      break;
    }
  }
  return 0;
}
