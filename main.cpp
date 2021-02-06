#include <iostream>
#include <sstream>
#include <time.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>

using namespace cv;
using namespace std;

int main(int argc, char* argv[]) {

	// Execution time
	double start,stop, dt;

  // Number of erosion/dilatation
  int niter = 3;

  // Create Background Subtractor objects
  Ptr<BackgroundSubtractor> pBackSub;
  // (history, varThreshold, detectShadows);
  pBackSub = createBackgroundSubtractorMOG2(500, 16, false);
  // pBackSub = createBackgroundSubtractorKNN(500, 16, false);

  // Create capture
  VideoCapture capture("video.mp4");
  capture.set(CAP_PROP_FRAME_WIDTH,  320);
  capture.set(CAP_PROP_FRAME_HEIGHT, 240);

  // Create window
  namedWindow("Raw", WINDOW_NORMAL);
  resizeWindow("Raw", 640*1.5, 320*1.5);

  if (!capture.isOpened()){
    cerr << "Unable to open file" << endl;
    return 0;
  }

  Mat frame, decision;
  while (true) {

    // Frame rate
    start = getTickCount();

    capture >> frame;
    if (frame.empty()) {
      break;
    }

    // Apply blur
    GaussianBlur(frame, frame, Size(11, 11), 0, 0);

    // Update the background model
    pBackSub->apply(frame, decision);

    // Erosion / Dilatation
    for (int i=0; i<niter; i++) {
      dilate(decision,decision,getStructuringElement(MORPH_ELLIPSE,Size(5,5)));
    }
    for (int i=0; i<niter; i++) {
      erode( decision,decision,getStructuringElement(MORPH_ELLIPSE,Size(5,5)));
    }

    // Trackbars
    createTrackbar("Niter", "Raw", &niter, 10);


    // Frame rate
    stop = getTickCount();
    dt = ((stop - start)/ getTickFrequency());
    putText(decision, to_string((int)(1/dt)) + "Hz", Point(30,30), FONT_HERSHEY_DUPLEX, 1.0, Scalar(255,255,255), 2);

    // Display result
    imshow("Raw", decision);

    int keyboard = waitKey(30);
    if (keyboard == 'q' || keyboard == 27) {
      break;
    }
  }
  return 0;
}
