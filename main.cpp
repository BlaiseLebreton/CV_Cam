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

// Resolution
#define WIDTH 852
#define HEIGHT 480

RNG rng(12345);

int   cmin    = 65;   // Minimum size of contour
int   cmax    = 1000; // Maximum size of contour
int   meanmin = 70;   // Minimum mean intensity of contour
int   niter   = 2;    // Number of erosion/dilatation
float distth  = 20;   // Distance threshold for new track
int   sky     = 110;  // Sky start
int   klost   = 20;   // Number of frame after a track is deleted if lost

// Structure of tracks
struct track {
  vector<Point2f> p; // Position
  Rect rect;         // Bounding box
  Point2f v;         // Speed
  Point2f a;         // Acceleration
  int kfound=0;      // Number of frame track has been updated
  int klost=0;       // Number of frame track has been lost
  bool found=true;   // Found corresponding object at frame
  Scalar color = Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255)); // Color
};
vector<track> Tracks;

// Structure of objects
struct object {
  Point2f p; // Position
  Rect rect; // Bounding box
};
vector<object> Objects;

// Distance function for data association
float Distance(object Obj, track Trck);

// Main thread
int main(int argc, char* argv[]) {

  // Execution time
  double start,stop, dt;

  // Create Background Subtractor objects
  Ptr<BackgroundSubtractor> pBackSub;
  pBackSub = createBackgroundSubtractorMOG2(500, 16, false);

  // Create capture
  VideoCapture capture("video.mp4");
  capture.set(CAP_PROP_FRAME_WIDTH,  WIDTH);
  capture.set(CAP_PROP_FRAME_HEIGHT, HEIGHT);

  // Create window
  namedWindow("Raw", WINDOW_KEEPRATIO);
  resizeWindow("Raw", WIDTH, HEIGHT);

  // Break if unable to open capture
  if (!capture.isOpened()){
    cerr << "Unable to open file" << endl;
    return 0;
  }

  // Images
  Mat raw, crop, filtered, decision, display;

  // Acquisition loop
  while (true) {

    // Frame rate
    start = getTickCount();

    // Get new frame
    capture >> raw;
    if (raw.empty()) {
      break;
    }

    // Resize
    resize(raw, raw, Size(WIDTH, HEIGHT));

    // Crop
    Rect roi = Rect(0, sky, raw.cols, raw.rows-sky-1);
    crop = raw(roi);

    // Create display image
    cvtColor(crop, display, COLOR_BGR2GRAY);
    cvtColor(display, display, COLOR_GRAY2BGR);

    // Apply gaussian blur
    GaussianBlur(crop, filtered, Size(11, 11), 0, 0);

    // Update the background model
    pBackSub->apply(filtered, decision);

    // Erosion / Dilatation
    for (int i=0; i<niter; i++) {
      dilate(decision,decision,getStructuringElement(MORPH_ELLIPSE,Size(3,3)));
    }
    for (int i=0; i<niter; i++) {
      erode( decision,decision,getStructuringElement(MORPH_ELLIPSE,Size(3,3)));
    }

    // // Show decision
    // cvtColor(decision, display, COLOR_GRAY2BGR);

    // Remove all old objects
    Objects.clear();

    // Find contours
    vector<vector<Point>> contours;
    findContours(decision,contours,RETR_EXTERNAL,CHAIN_APPROX_NONE);


    // Filter contours
    Rect boundRect;
    for( size_t i = 0; i < contours.size(); i++ ) {
      boundRect = boundingRect( contours[i] );
      int m = (int)sum(mean(decision(boundRect)))[0];

      // If an object fits the parameters
      if ((contours[i].size() > cmin) && (contours[i].size() < cmax) && (m > meanmin)) {

        // Draw the contour of the detected object
        drawContours(display, contours, i, Scalar(255,255,0), 1);

        // Save it
        Objects.push_back({
          (boundRect.tl() + boundRect.br())/2.0,
          boundRect
        });
      }
    }

    for (int trck = 0; trck < Tracks.size(); trck++) {
      // Initialize it to not found
      Tracks[trck].found = false;

      // Predict new position of tracks
      Tracks[trck].a = Tracks[trck].a;
      Tracks[trck].v = Tracks[trck].v + 1/2*Tracks[trck].a;
      Tracks[trck].p.push_back(Tracks[trck].p.back() + Tracks[trck].v);

      // Draw the new estimated position
      circle(display, Tracks[trck].p.back(), 1, Scalar(0,255,0), 2);
    }

    // Data association
    for (int obj = 0; obj < Objects.size(); obj++) {

      // Draw objects
      rectangle(display, Objects[obj].rect.tl(), Objects[obj].rect.br(), Scalar(0,0,255), 1);

      float mindist = 1000000;
      int mintrck = -1;
      float dist;
      for (int trck = 0; trck < Tracks.size(); trck++) {

        // Measure similarity
        dist = Distance(Objects[obj], Tracks[trck]);

        // Save the one with the minimum distance
        if (dist < mindist) {
          mindist = dist;
          mintrck = trck;
        }
      }

      // Create new track if no track match the object
      if ((mindist > distth) || (mintrck == -1)) {
        track NewElem;
        NewElem.p.push_back(Objects[obj].p);
        NewElem.rect = Objects[obj].rect;
        Tracks.push_back(NewElem);
      }
      else {
        // Update track with object
        Tracks[mintrck].kfound++;
        Tracks[mintrck].found = true;
        Tracks[mintrck].p.back() = Objects[obj].p; // TODO : Bayesian filter

        // Calculate new speed and acceleration
        Tracks[mintrck].v = Tracks[mintrck].p.end()[-1] - Tracks[mintrck].p.end()[-2];
        Tracks[mintrck].a = (Tracks[mintrck].p.end()[-1] - Tracks[mintrck].p.end()[-2]) - (Tracks[mintrck].p.end()[-2] - Tracks[mintrck].p.end()[-3]);
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
      if (Tracks[trck].klost <= klost || Tracks[trck].found) {
        Tracks2.push_back(Tracks[trck]);
      }
    }
    Tracks = Tracks2;

    // Draw tracks
    for (int trck = 0; trck < Tracks.size(); trck++) {
      circle     (display, Tracks[trck].p.back(), 1, Tracks[trck].color, 2);
      arrowedLine(display, Tracks[trck].p.back(), Tracks[trck].p.back() + Tracks[trck].v*10, Tracks[trck].color, 1, 8, 0, 0.1);
      // putText    (display, to_string(trck),                Tracks[trck].p.back() + Point2f(0,0),  FONT_HERSHEY_DUPLEX, 0.5, Scalar(255,255,255), 1);
      // putText    (display, to_string(Tracks[trck].kfound), Tracks[trck].p.back() + Point2f(0,10), FONT_HERSHEY_DUPLEX, 0.5, Scalar(255,255,255), 1);
      // putText    (display, to_string(Tracks[trck].klost),  Tracks[trck].p.back() + Point2f(0,20), FONT_HERSHEY_DUPLEX, 0.5, Scalar(255,255,255), 1);
      for (int ip = 1; ip < Tracks[trck].p.size(); ip++) {
        line(display, Tracks[trck].p[ip-1], Tracks[trck].p[ip], Tracks[trck].color, 2);
      }
    }

    // Trackbars
    createTrackbar("Niter", "Raw", &niter, 10);
    createTrackbar("Sky", "Raw", &sky, raw.rows);
    createTrackbar("Min Size", "Raw", &cmin, 1000);
    createTrackbar("Max Size", "Raw", &cmax, 1000);
    createTrackbar("Min Mean", "Raw", &meanmin, 1000);

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
