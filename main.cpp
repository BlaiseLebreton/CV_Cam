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

#define WIDTH 852
#define HEIGHT 480


RNG rng(12345);


int cmin = 50, cmax = 1000; // maxi/mini longeur contour
int niter  = 3; // Number of erosion/dilatation
float distth = 20; // Distance threshold for new track
int sky    = 110; // Sky start

struct track {
  Point2f p; // Position
  Point2f v; // Speed
  Point2f a; // Acceleration
  Point2f op; // Old position
  Point2f ov; // Old position
  int iter=0; // Number of time track has been updated
  // ??
};
vector<track> Tracks;

struct object {
  Point2f p; // Position
  int width;
  int height;
  Rect rect;
};

vector<object> Objects;

// static void arrowedLine(InputOutputArray img, Point pt1, Point pt2, const Scalar& color, int thickness=1, int line_type=8, int shift=0, double tipLength=0.1);


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
  VideoCapture capture("video2.mp4");
  capture.set(CAP_PROP_FRAME_WIDTH,  WIDTH);
  capture.set(CAP_PROP_FRAME_HEIGHT, HEIGHT);

  // Create window
  namedWindow("Raw", WINDOW_KEEPRATIO);
  // resizeWindow("Raw", WIDTH, HEIGHT);

  if (!capture.isOpened()){
    cerr << "Unable to open file" << endl;
    return 0;
  }

  Mat raw, crop, filtered, decision, display;
  while (true) {

    // Frame rate
    start = getTickCount();

    capture >> raw;
    if (raw.empty()) {
      break;
    }

    // Resize
    resize(raw, raw, Size(WIDTH, HEIGHT));
    // Crop
    Rect roi = Rect(0, sky, raw.cols, raw.rows-sky-1);
    crop = raw(roi);

    // Creating display image
    cvtColor(crop, display, COLOR_BGR2GRAY);
    cvtColor(display, display, COLOR_GRAY2BGR);

    // Apply blur
    GaussianBlur(crop, filtered, Size(31, 31), 0, 0);

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
    vector<Rect> boundRect(contours.size());


    // Elimination des anciens objets
    Objects.clear();

    for( size_t i = 0; i < contours.size(); i++ ) {
      if ((contours[i].size() > cmin) && (contours[i].size() < cmax)) {
        boundRect[i] = boundingRect( contours[i] );
        Objects.push_back({
          (boundRect[i].tl() + boundRect[i].br())/2.0, // Position
          boundRect[i].width,
          boundRect[i].height,
          boundRect[i]
        });
      }
    }

    // drawContours(display, contours, -1, Scalar(0,0,255), 2);

    // Predict tracks
    for (int trck = 0; trck < Tracks.size(); trck++) {
      Tracks[trck].op = Tracks[trck].p;
      Tracks[trck].ov = Tracks[trck].v;

      Tracks[trck].v = Tracks[trck].v + 1/2*Tracks[trck].a;
      Tracks[trck].p = Tracks[trck].p + Tracks[trck].v;
      circle(display, Tracks[trck].p, 2, Scalar(255,0,0), 2);
    }

    // Data association
    for (int obj = 0; obj < Objects.size(); obj++) {

      // Draw objects
      // circle(display, Objects[obj].p, 2, Scalar(255,0,0), 2);
      rectangle(display, Objects[obj].rect.tl(), Objects[obj].rect.br(), Scalar(0,0,255), 2 );

      float mindist = 1000000;
      int mintrck = -1;
      float dist;
      for (int trck = 0; trck < Tracks.size(); trck++) {
        // Measure similarity
        Point2f dx = Objects[obj].p - Tracks[trck].p;
        dist = sqrt(abs(dx.x*dx.x + dx.y*dx.y));

        if (dist < mindist) {
          mindist = dist;
          mintrck = trck;
        }
      }

      // Create new track
      printf("Obj : %d | Track : %d | dist = %f\n", obj,mintrck,mindist);
      if ((mindist > distth) || (mintrck == -1)) {
        printf("Creating new track for %d\n",obj);
        Tracks.push_back({
          Objects[obj].p, // Position
          Point2f(0,0), // Speed
          Point2f(0,0), // Acceleration
        });

      }
      else {
        // Update track with object
        printf("Found track %d for %d\n",mintrck,obj);
        Tracks[mintrck].iter++;

        Point2f dx = Objects[obj].p - Tracks[mintrck].op;
        Point2f dv = dx;
        Point2f da = (dv - Tracks[mintrck].ov);

        Tracks[mintrck].p = Objects[obj].p; // Filtre bayésien ?
        Tracks[mintrck].v = dv;
        Tracks[mintrck].a = da;

      }
    }
    if (Objects.size() > 0) {
      printf("\n");
    }

    // Draw tracks
    for (int trck = 0; trck < Tracks.size(); trck++) {
      circle(display, Tracks[trck].p, 2, Scalar(0,255,0), 2);
      arrowedLine(display, Tracks[trck].p, Tracks[trck].p + Tracks[trck].v*10, Scalar(0,255,0), 1, 8, 0, 0.1);
    }

    // Trackbars
    createTrackbar("Niter", "Raw", &niter, 10);
    createTrackbar("Sky", "Raw", &sky, raw.rows);
    createTrackbar("Min Size", "Raw", &cmin, 1000);
    createTrackbar("Max Size", "Raw", &cmax, 1000);


    // Frame rate
    stop = getTickCount();
    dt = ((stop - start)/ getTickFrequency());
    putText(display, to_string((int)(1/dt)) + "Hz", Point(50,50), FONT_HERSHEY_DUPLEX, 2.0, Scalar(255,255,255), 2);

    // Display result
    imshow("Raw", display);

    int keyboard = waitKey(1);
    if (keyboard == 'q' || keyboard == 27) {
      break;
    }
  }
  return 0;
}

//
// static void arrowedLine(InputOutputArray img, Point pt1, Point pt2, const Scalar& color, int thickness=1, int line_type=8, int shift=0, double tipLength=0.1)
// {
//     const double tipSize = norm(pt1-pt2)*tipLength; // Factor to normalize the size of the tip depending on the length of the arrow
//     line(img, pt1, pt2, color, thickness, line_type, shift);
//     const double angle = atan2( (double) pt1.y - pt2.y, (double) pt1.x - pt2.x );
//     Point p(cvRound(pt2.x + tipSize * cos(angle + CV_PI / 4)),
//     cvRound(pt2.y + tipSize * sin(angle + CV_PI / 4)));
//     line(img, p, pt2, color, thickness, line_type, shift);
//     p.x = cvRound(pt2.x + tipSize * cos(angle - CV_PI / 4));
//     p.y = cvRound(pt2.y + tipSize * sin(angle - CV_PI / 4));
//     line(img, p, pt2, color, thickness, line_type, shift);
// }
