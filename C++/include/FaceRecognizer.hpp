/**
 * @file      FaceRecognizer.hpp
 * @author    @ZouariOmar (zouariomar20@gmail.com)
 * @brief     FaceRecognizer header file
 * @version   0.1
 * @date      2025-04-12
 * @copyright Copyright (c) 2025
 * @link https://github.com/ZouariOmar/Astra/project/inc/FaceRecognizer.hpp
 * FaceRecognizer.hpp @endlink
 */

//? Pre-Processor prototype declaration part
#ifndef __FACE_RECOGNITION_HPP__
#define __FACE_RECOGNITION_HPP__
#define __FACES_DEFAULT_DIR__ "../Faces"
#define __CASCADE_DEFAULT_MODEL__ "../Models/haarcascade_frontalface_default.xml"
#define __FACE_RECOGNIZER_DEFAULT_MODEL__ "../Models/face_recognizer.yaml"
#define __CAMERA_DEFAULT_FPS__ 30 // ~33 times per second - 30 milliseconds
#define __CONFIDENCE_DEFAULT_SCORE__ 0.2

//? Include prototype declaration part
//* Include std c++ header(s)
#include <map>

//* Include std headers (Qt)
#include <QtCore/QObject>
#include <QtWidgets/QLabel>

//* Include OpenCV header(s)
#include <opencv2/face.hpp>      // For LBPHFaceRecognizer
#include <opencv2/objdetect.hpp> // For HaarCascade
#include <opencv2/opencv.hpp>

//? Class prototype declaration part
class FaceRecognizer {
public:
  ~FaceRecognizer();
  void trainEmbeddedModel(const std::string &, const std::string &, const std::string &);
  void captureFrame();
  void captureFrame(QLabel *);
  std::string recognize(QLabel *);
  void load();

private:
  cv::VideoCapture cap;                             // Video capture object to access the camera
  cv::Mat currentFrame;                             // Holds the current captured frame
  cv::CascadeClassifier faceCascade;                // Haar Cascade for face detection
  cv::Ptr<cv::face::LBPHFaceRecognizer> recognizer; // LBPH face recognizer
  std::map<int, std::string> indexer;               // Map to store labels and corresponding names

  QImage matToQImage(const cv::Mat &mat) const;
  std::pair<std::vector<cv::Mat>, std::vector<int>>
  loadData(const std::string &dataDir);
  void loadIndexer(const std::string &dataDir);
  void loadEmbeddedModel(const std::string &);
  void loadCascadeModel(const std::string &);
}; // FaceRecognizer class

#endif // __FACE_RECOGNITION_HPP__
