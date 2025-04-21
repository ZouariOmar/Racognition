/**
 * @file      FaceRecognizer.cpp
 * @author    @ZouariOmar (zouariomar20@gmail.com)
 * @brief     FaceRecognizer source file
 * @version   0.1
 * @date      2025-04-12
 * @copyright Copyright (c) 2025
 * @link https://github.com/ZouariOmar/Astra/project/src/FaceRecognizer.cpp FaceRecognizer.cpp @endlink
 */

//? Include prototype declaration part
#include "../include/FaceRecognizer.hpp"

//* Include std c++ header(s)
#include <filesystem>

//* Include std headers (Qt)
#include <QtGui/QImage>
#include <QtGui/QPixmap>

//? Function/Class prototype dev part

/**
 * @fn    FaceRecognizer::~FaceRecognizer()
 * @brief Destroy the FaceRecognizer::FaceRecognizer object
 */
FaceRecognizer::~FaceRecognizer() {
  recognizer.release(); // Must release it before releasing `cap`
  if (cap.isOpened())
    cap.release();
}

/**
 * @fn     FaceRecognizer::load()
 * @brief  load all FaceRecognizer loaders (Indexer, CascadeModel, EmbeddedModel)
 * @return void
 */
void FaceRecognizer::load() {
  loadIndexer(__FACES_DEFAULT_DIR__); //! Security issue (`X` can change folders //! names and make spoofing attack)
  loadCascadeModel(__CASCADE_DEFAULT_MODEL__);
  loadEmbeddedModel(__FACE_RECOGNIZER_DEFAULT_MODEL__);
}

/**
 * @fn                     FaceRecognizer::loadCascadeModel(const std::string &)
 * @brief                  Load "Cascade Model"
 * @param cascadeModelPath {const std::string &}
 * @return                 void
 */
void FaceRecognizer::loadCascadeModel(const std::string &cascadeModelPath) {
  try {
    if (!faceCascade.load(cascadeModelPath)) {
      std::cerr << "[CASCADE MODEL] Failed to load Cascade MODEL!\n";
      return;
    }
    std::cerr << "[CASCADE MODEL] Cascade Model loaded successfully from " << cascadeModelPath << '\n';
  } catch (const cv::Exception &e) {
    std::cerr << "[CASCADE MODEL] Cascade Model loading error: " << e.what() << '\n';
  }
}

/**
 * @fn                      FaceRecognizer::loadEmbeddedModel(const std::string &)
 * @brief                   Load "Embedded Model"
 * @param embeddedModelPath {const std::string &}
 * @return                  void
 */
void FaceRecognizer::loadEmbeddedModel(const std::string &embeddedModelPath) {
  try {
    recognizer = cv::face::LBPHFaceRecognizer::create();
    recognizer->read(embeddedModelPath);
    std::cout << "[FaceRecognizer] Embedded Model loaded successfully from " << embeddedModelPath << '\n';
  } catch (const cv::Exception &e) {
    std::cerr << "[FaceRecognizer] Embedded Model loading error: " << e.what() << '\n';
  }
}

/**
 * @fn            FaceRecognizer::loadIndexer(const std::string &)
 * @brief         Load labels and corresponding names in `indexer`
 * @param dataDir {const std::string &}
 * @return        void
 */
void FaceRecognizer::loadIndexer(const std::string &dataDir) {
  int label{};
  for (const auto &entry : std::filesystem::directory_iterator(dataDir))
    if (entry.is_directory()) {
      std::string personName = entry.path().filename().string();
      std::cout << "Processing folder:" << personName << '\n';
      indexer[label] = personName, label++;
    }
}

//! sudo chmod 666 /dev/video0
std::string FaceRecognizer::recognize(QLabel *cameraLabel) {
  if (!cap.isOpened()) // Open the camera once at the start
    if (!cap.open(0, cv::CAP_V4L2)) {
      std::cerr << "Error: Cannot open camera!" << '\n';
      return "";
    }

  cap >> currentFrame;
  if (currentFrame.empty()) {
    std::cerr << "Error: Empty frame captured!" << '\n';
    return "";
  }

  cv::Mat gray;
  if (currentFrame.channels() == 3)
    cv::cvtColor(currentFrame, gray, cv::COLOR_BGR2GRAY);
  else
    gray = currentFrame;

  std::vector<cv::Rect> faces;
  faceCascade.detectMultiScale(gray, faces, 1.1, 5, 0, cv::Size(100, 100));

  if (!faces.empty()) {
    cv::Mat faceROI = gray(faces[0]);

    int label(-1);
    double confidence{};
    recognizer->predict(faceROI, label, confidence);

    confidence = 1.0 - (confidence / 100.0);
    std::string username(indexer[label]);

    if (confidence >= __CONFIDENCE_DEFAULT_SCORE__) {
      std::cout << "[username] " << username << " ### [confidence] " << confidence << '\n';
      return username; // Return user information
    } else
      std::cerr << "No confident match found (face detected)!" << '\n';
  } else
    std::cerr << "No confident match found!" << '\n';

  // Convert OpenCV Mat to QImage
  cameraLabel->setPixmap(QPixmap::fromImage(matToQImage(currentFrame).scaled(cameraLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation)));
  return ""; // Not found
}

/**
 * @brief             Capture the frame from the camera
 * @details           Use it only in test mode (only show the recognized face with username label)
 * @param cameraLabel {QLabel *}
 * @return            void
 */
void FaceRecognizer::captureFrame(QLabel *cameraLabel) {
  if (!cap.isOpened())
    if (!cap.open(0, cv::CAP_V4L2))
      return;

  cap >> currentFrame;
  if (currentFrame.empty()) {
    std::cerr << "Error: Empty frame captured!" << '\n';
    return;
  }

  cv::Mat gray;
  if (currentFrame.channels() == 3)
    cv::cvtColor(currentFrame, gray, cv::COLOR_BGR2GRAY);
  else
    gray = currentFrame;

  std::vector<cv::Rect> faces;
  faceCascade.detectMultiScale(gray, faces);

  if (!faces.empty()) {
    cv::Mat faceROI = gray(faces[0]);

    int label(-1);
    double confidence{};
    recognizer->predict(faceROI, label, confidence);

    confidence = 1.0 - (confidence / 100.0);
    std::string name(indexer[label]);
    // cv::imwrite("test" + std::to_string(confidence) + ".png", faceROI);

    if (confidence >= __CONFIDENCE_DEFAULT_SCORE__) {
      std::cout << "[username] " << name << " ### [confidence] " << confidence
                << '\n';
      cv::putText(currentFrame, name + " - " + std::to_string(confidence),
                  faces[0].tl(), cv::FONT_HERSHEY_SIMPLEX, 1,
                  cv::Scalar(255, 0, 0), 2);
      cv::rectangle(currentFrame, faces[0], cv::Scalar(255, 0, 0), 2);
    } else
      std::cerr << "No confident match found (face detected)!" << '\n';
  } else
    std::cerr << "No confident match found!" << '\n';

  cameraLabel->setPixmap(
      QPixmap::fromImage(matToQImage(currentFrame)
                             .scaled(cameraLabel->size(), Qt::KeepAspectRatio,
                                     Qt::SmoothTransformation)));
}

/**
 * @fn     FaceRecognizer::captureFrame()
 * @brief  Capture the frame from OpenCV camera
 * @retrun void
 */
void FaceRecognizer::captureFrame() {
  if (!cap.isOpened()) {
    if (!cap.open(0, cv::CAP_V4L2)) {
      std::cerr << "Error: Could not open camera!" << std::endl;
      return;
    }
  }
  while (1) {
    // Capture a single frame
    cap >> currentFrame;
    if (currentFrame.empty()) {
      std::cerr << "Error: Empty frame captured!\n";
      break;
    }

    // Convert to grayscale
    cv::Mat gray;
    if (currentFrame.channels() == 3)
      cv::cvtColor(currentFrame, gray, cv::COLOR_BGR2GRAY);
    else
      gray = currentFrame;

    // Detect faces
    std::vector<cv::Rect> faces;
    faceCascade.detectMultiScale(gray, faces);
    if (!faces.empty()) {
      cv::Mat faceROI = gray(faces[0]);
      int label(-1);
      double confidence(0.0);
      recognizer->predict(faceROI, label, confidence);
      confidence = 1.0 - (confidence / 100.0);
      std::string name(indexer[label]);

      if (confidence >= __CONFIDENCE_DEFAULT_SCORE__) {
        std::cout << "[username] " << name
                  << " ### [confidence] " << confidence << std::endl;

        // Draw on frame
        cv::putText(currentFrame, name + " - " + std::to_string(confidence),
                    faces[0].tl(), cv::FONT_HERSHEY_SIMPLEX, 1,
                    cv::Scalar(0, 255, 0), 2);
        cv::rectangle(currentFrame, faces[0], cv::Scalar(0, 255, 0), 2);
      } else
        std::cerr << "No confident match found (face detected)!\n";
    } else
      std::cerr << "No face detected!" << std::endl;

    // Show the result frame
    cv::imshow("Face Recognition", currentFrame);

    // Press `ESC` to exit
    if (cv::waitKey(1) == 27) {
      std::cout << "Exiting recognition loop...\n";
      cap.release();
      cv::destroyAllWindows();
    }
  }
}

/**
 * @fn        FaceRecognizer::matToQImage(const cv::Mat &)
 * @brief     Convert `cv::Mat` to `QImage`
 * @param mat {const cv::Mat &}
 * @return    QImage
 */
QImage FaceRecognizer::matToQImage(const cv::Mat &mat) const {
  QImage img(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_RGB888);
  return img.rgbSwapped();
}

/**
 * @fn            FaceRecognizer::loadData(const std::string &, const
 * FaceRecognizerFlags &)
 * @brief         Load training data
 * @param dataDir {const std::string &}
 * @param flag    {const FaceRecognizerFlags &}
 * @return        std::pair<std::vector<cv::Mat>, std::vector<int>>
 */
std::pair<std::vector<cv::Mat>, std::vector<int>>
FaceRecognizer::loadData(const std::string &dataDir) {
  std::vector<cv::Mat> images;
  std::vector<int> labels;
  int label{}, counter{};

  for (const auto &entry : std::filesystem::directory_iterator(dataDir)) {
    if (entry.is_directory()) {
      std::string personName = entry.path().filename().string();
      qDebug() << "Processing folder:" << QString::fromStdString(personName);

      for (const auto &videoEntry :
           std::filesystem::directory_iterator(entry.path())) {
        if (videoEntry.is_regular_file()) {
          std::string videoPath = videoEntry.path().string();
          cv::VideoCapture cap(videoPath);
          if (!cap.isOpened()) {
            std::cerr << "Could not open video: " << videoPath << '\n';
            continue;
          }

          cv::Mat frame;
          while (cap.read(frame)) {
            if (frame.empty())
              continue;

            cv::Mat gray;
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
            std::vector<cv::Rect> faces;
            faceCascade.detectMultiScale(gray, faces);

            for (const cv::Rect &face : faces) {
              cv::Mat faceROI = gray(face);
              images.push_back(faceROI), labels.push_back(label), counter++;
            }
          }
          cap.release();
        }
      }
      label++;
    }
  }

  std::cout << "Total Detected Faces from All Frames: " << counter << '\n';
  return {images, labels};
}

/**
 * @fn                      FaceRecognizer::trainModel(const std::string &, const std::string &)
 * @brief                   Train a model with `dataLoadPath` and save it in `modelSavePath`
 * @param modelSavePath     {const std::string &}
 * @param dataLoadPath      {const std::string &}
 * @param cascadeModelPath  {const std::string &}
 * @return                  void
 */
void FaceRecognizer::trainEmbeddedModel(const std::string &modelSavePath,
                                        const std::string &cascadeModelPath,
                                        const std::string &dataLoadPath) {
  recognizer = cv::face::LBPHFaceRecognizer::create();
  loadCascadeModel(cascadeModelPath);
  auto [images, labels] = loadData(dataLoadPath);
  if (!images.empty() && !labels.empty()) {
    recognizer->train(images, labels), recognizer->save(modelSavePath);
    std::cout << "Training complete and model saved!\n";
  } else
    std::cerr << "No faces found for training!\n";
}
