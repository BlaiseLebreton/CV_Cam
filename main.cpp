#include "opencv2/opencv.hpp"
#include <iostream>

using namespace std;
using namespace cv;

int seuil = 25;
int niter = 3;
int cmin = 0, cmax = 10000; // maxi/mini longeur contour

int main(){
  VideoCapture cap("video.mp4");

  if(!cap.isOpened()){
    cout << "Error opening video stream or file" << endl;
    return -1;
  }
  cap.set(CAP_PROP_FRAME_WIDTH,  320);
  cap.set(CAP_PROP_FRAME_HEIGHT, 240);

  namedWindow("Raw", WINDOW_NORMAL);
  resizeWindow("Raw", 640*2, 320*2);

  Mat new_frameRGB;
  Mat new_frame;
  Mat old_frame;
  Mat decision;
  Mat display;

  while(1) {
    // printf("Loop\n");
    cap >> new_frameRGB;
    new_frame = new_frameRGB;
    cvtColor(new_frame, new_frame, COLOR_BGR2GRAY);


    if (new_frame.empty()) {
      break;
    }
    if (old_frame.empty()) {
      old_frame = new_frame;
    }

    decision = (abs(new_frame - old_frame) > seuil)*255;

    for (int i=0; i<niter; i++) {
      dilate(decision,decision,getStructuringElement(MORPH_ELLIPSE,Size(5,5)));
    }
    for (int i=0; i<niter; i++) {
      erode( decision,decision,getStructuringElement(MORPH_ELLIPSE,Size(5,5)));
    }

    // Contour
    vector<vector<Point> > contours;
    findContours(decision,contours,RETR_EXTERNAL,CHAIN_APPROX_NONE);

    // Elimination des contours trop long/court
    vector<vector<Point> >::iterator itc = contours.begin();
    vector<vector<Point> >::iterator i_m = itc;
    int max = 0;

    // cvtColor(decision, decision, COLOR_GRAY2BGR);
  	while (itc != contours.end()) {

  		if (itc->size() < cmin || itc->size() > cmax)
  			itc = contours.erase(itc);
  		else
      {
        if (itc->size() > max){
          max = itc->size();
          i_m = itc;
        }
        Moments mom = moments(*itc);

        // Centre de gravite
        circle(new_frameRGB, Point(mom.m10/mom.m00,mom.m01/mom.m00), 2, Scalar(0,255,0), 2);
        ++itc;
      }
  	}
    drawContours(new_frameRGB, contours, -1, Scalar(0,0,255), 2);

    createTrackbar("Seuil", "Raw", &seuil, 255);
    createTrackbar("Niter", "Raw", &niter, 10);
    createTrackbar("Min Size", "Raw", &cmin, 10000);
    createTrackbar("Max Size", "Raw", &cmax, 10000);
    // hconcat(new_frame, decision, display);
    imshow("Raw", new_frameRGB);
    // imshow("Raw", decision);

    old_frame = new_frame;

    char c=(char)waitKey(25);
    if(c==27) {
      break;
    }
  }

  cap.release();
  destroyAllWindows();

  return 0;
}
