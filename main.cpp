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
  vector<Point2f> p; // Position
  Rect rect; // Bounding box
  Point2f v; // Speed
  Point2f a; // Acceleration
  Point2f op; // Old position
  Point2f ov; // Old position
  int kfound=0; // Number of time track has been updated
  int klost=0; // Number of time track has been updated
  bool found=true; // Number of time track has been updated
  vector<Point2f> hp; // History of positions
  Scalar color = Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255)); // Color
};
vector<track> Tracks;

struct object {
  Point2f p; // Position
  Rect rect;
};

vector<object> Objects;

float Distance(object Obj, track Trck);


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
  resizeWindow("Raw", WIDTH, HEIGHT);
  // Create window
  // namedWindow("Decision", WINDOW_KEEPRATIO);
  // resizeWindow("Decision", WIDTH, HEIGHT);

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
    GaussianBlur(crop, filtered, Size(11, 11), 0, 0);

    // Update the background model
    pBackSub->apply(filtered, decision);
    // cvtColor(decision, display, COLOR_GRAY2BGR);

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
          boundRect[i]
        });
      }
    }

    // Predict tracks
    for (int trck = 0; trck < Tracks.size(); trck++) {
      Tracks[trck].found = false;
      // Tracks[trck].op = Tracks[trck].p;
      Tracks[trck].ov = Tracks[trck].v;

      Tracks[trck].v = Tracks[trck].v + 1/2*Tracks[trck].a;
      Tracks[trck].p.push_back(Tracks[trck].p.back() + Tracks[trck].v);
      circle(display, Tracks[trck].p.back(), 2, Scalar(0,255,0), 2);
    }

    // Data association
    for (int obj = 0; obj < Objects.size(); obj++) {

      // Draw objects
      rectangle(display, Objects[obj].rect.tl(), Objects[obj].rect.br(), Scalar(0,0,255), 2 );

      float mindist = 1000000;
      int mintrck = -1;
      float dist;
      for (int trck = 0; trck < Tracks.size(); trck++) {
        // Measure similarity
        dist = Distance(Objects[obj], Tracks[trck]);

        if (dist < mindist) {
          mindist = dist;
          mintrck = trck;
        }
      }

      // Create new track
      if ((mindist > distth) || (mintrck == -1)) {
        track NewElem;
        NewElem.p.push_back(Objects[obj].p);
        NewElem.rect = Objects[obj].rect;
        Tracks.push_back(NewElem);

        if (Tracks.size() > 5) {
          break;
        }

      }
      else {
        // Update track with object
        Tracks[mintrck].kfound++;
        Tracks[mintrck].found = true;

        // Calculate speed and acceleration
        Point2f dx = Objects[obj].p - Tracks[mintrck].p.end()[-2];
        Point2f dv = dx;
        Point2f da = (dv - Tracks[mintrck].ov);

        // Update position, speed and acceleration
        Tracks[mintrck].p.back() = Objects[obj].p;
        Tracks[mintrck].v = dv;
        Tracks[mintrck].a = da;

      }
    }

    // Update counts
    for (int trck = 0; trck < Tracks.size(); trck++) {
      if (!Tracks[trck].found) {
        Tracks[trck].klost++;
        Tracks[trck].kfound = 0;   // Pas sur
      }
      else {
        Tracks[trck].klost = 0;
        Tracks[trck].kfound++;   // Pas sur
      }
    }

    // Resample
    vector<track> Tracks2;
    for (int trck = 0; trck < Tracks.size(); trck++) {
      if (Tracks[trck].klost <= 10 || Tracks[trck].found) {
        Tracks2.push_back(Tracks[trck]);
      }
    }
    Tracks = Tracks2;

    // Draw tracks
    for (int trck = 0; trck < Tracks.size(); trck++) {
      circle     (display, Tracks[trck].p.back(), 2, Tracks[trck].color, 2);
      arrowedLine(display, Tracks[trck].p.back(), Tracks[trck].p.back() + Tracks[trck].v*10, Tracks[trck].color, 1, 8, 0, 0.1);
      putText    (display, to_string(trck),                Tracks[trck].p.back() + Point2f(0,0),  FONT_HERSHEY_DUPLEX, 0.5, Scalar(255,255,255), 2);
      putText    (display, to_string(Tracks[trck].kfound), Tracks[trck].p.back() + Point2f(0,10), FONT_HERSHEY_DUPLEX, 0.5, Scalar(255,255,255), 2);
      putText    (display, to_string(Tracks[trck].klost),  Tracks[trck].p.back() + Point2f(0,20), FONT_HERSHEY_DUPLEX, 0.5, Scalar(255,255,255), 2);
      for (int ip = 1; ip < Tracks[trck].p.size(); ip++) {
        line(display, Tracks[trck].p[ip-1], Tracks[trck].p[ip], Tracks[trck].color);
      }
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
    // imshow("Decision", decision);

    // Break if q or esc is pressed
    int keyboard = waitKey(30);
    if (keyboard == 'q' || keyboard == 27) {
      break;
    }
  }
  return 0;
}

// Distance function
float a1 = 1.0;
float a2 = 0.25;
float a3 = 0.25;
float Distance(object Obj, track Trck) {
  float d = 0;
  Point2f dcr = Obj.p - Trck.p.back();
  // Point2f dtl = Obj.rect.tl() - Trck.rect.tl();
  // Point2f dbr = Obj.rect.br() - Trck.rect.br();
  d += a1*sqrt(abs(dcr.x*dcr.x + dcr.y*dcr.y));
  // d += a2*sqrt(abs(dtl.x*dtl.x + dtl.y*dtl.y));
  // d += a3*sqrt(abs(dbr.x*dbr.x + dbr.y*dbr.y));

  return d;
}
