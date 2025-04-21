/**
 * @file    main.cpp
 * @author  @ZouariOmar (zouariomar20@gmail.com)
 * @brief   main source file
 * @version 0.1
 * @date    2025-04-18
 * @copyright Copyright (c) 2025
 * @link https://github.com/ZouariOmar ZouariOmar @endlink
 */

//? Include prototype declaration part
#include "../include/FaceRecognizer.hpp"

//* Include std c++ header(s)
#include <cstdlib>

//? Main function prototype dev part

/**
 * @fn     main()
 * @brief  Main lancer file
 * @return int
 */
int main(void) {
  FaceRecognizer fr;

  // Recgnize mode (using opencv camera)
  // fr.load();
  // fr.captureFrame();

  //* Training mode
  // fr.trainEmbeddedModel(__FACE_RECOGNIZER_DEFAULT_MODEL__, __CASCADE_DEFAULT_MODEL__, __FACES_DEFAULT_DIR__);

  return EXIT_SUCCESS;
}
