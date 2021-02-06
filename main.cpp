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

struct track {
  Point p; // Position
  Point v; // Speed
  Point a; // Acceleration
  // ??
};
vector<track> Tracks;
int nt = 0;

struct object {
  Point p; // Position
  // ??
};

vector<track> Objects;
int no = 0;

int main(int argc, char* argv[]) {

  int x,y;
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
    findContours(decision,contours,RETR_EXTERNAL,CHAIN_APPROX_NONE);

    // Elimination des contours trop long/court
    vector<vector<Point>>::iterator itc = contours.begin();

    // Elimination des anciens objets
    Objects.clear();
    no = 0;

    while (itc != contours.end()) {

      if (itc->size() < cmin || itc->size() > cmax) {
        itc = contours.erase(itc);
      }
      else {
        // Centre de gravite
        Moments mom = moments(*itc);

        Objects.push_back({
          Point(mom.m10/mom.m00,mom.m01/mom.m00), // Position
        });

        no++;

        ++itc;
      }
    }
    // drawContours(display, contours, -1, Scalar(0,0,255), 2);

    // Data association
    for (int obj = 0; obj < no; obj++) {

      circle(display, Objects[obj].p, 2, Scalar(0,0,255), 2);

      for (int trck = 0; trck < nt; trck++) {

      }
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
