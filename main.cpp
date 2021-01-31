#include "opencv2/opencv.hpp"
#include <iostream>

using namespace std;
using namespace cv;

int main(){
  VideoCapture cap("video.mp4");

  if(!cap.isOpened()){
    cout << "Error opening video stream or file" << endl;
    return -1;
  }

  namedWindow("Raw", WINDOW_NORMAL);
  resizeWindow("Raw", 640, 640);

  while(1){

    Mat frame;
    cap >> frame;

    if (frame.empty())
      break;

    imshow("Raw", frame);

    char c=(char)waitKey(25);
    if(c==27)
      break;
  }

  cap.release();
  destroyAllWindows();

  return 0;
}
