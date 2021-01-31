#include "opencv2/opencv.hpp"
#include <iostream>

using namespace std;
using namespace cv;

void onTrackbar_changed(int, void*);
int seuil = 50;

int main(){
  VideoCapture cap("video.mp4");

  if(!cap.isOpened()){
    cout << "Error opening video stream or file" << endl;
    return -1;
  }
  cap.set(CAP_PROP_FRAME_WIDTH,  320);
  cap.set(CAP_PROP_FRAME_HEIGHT, 240);

  namedWindow("Raw", WINDOW_NORMAL);
  resizeWindow("Raw", 640, 640);

  Mat new_frame;
  Mat old_frame;
  Mat decision;
  while(1) {
    // printf("Loop\n");
    cap >> new_frame;

    cvtColor(new_frame, new_frame, COLOR_BGR2GRAY);

    if (new_frame.empty()) {
      break;
    }
    if (old_frame.empty()) {
      old_frame = new_frame;
    }

    decision = (abs(new_frame - old_frame) > seuil)*255;

    createTrackbar("Seuil", "Raw", &seuil, 100, onTrackbar_changed);
    imshow("Raw", decision);

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

void onTrackbar_changed(int, void*) {

}
