#!/bin/bash
# ====================================================================================
# Script: run.sh
# Purpose: This script automates the build, clean, and run processes
#          for a project using CMake and a specified build tool (make
#          or ninja). It allows users to perform operations like
#          configuring the project, building, cleaning the build
#          directory, and running the application.
# Usage: ./run.sh <command> [options]
#
# Arguments:
#   <command>       - The action to perform. Supported commands:
#                       build | run | clean | <-h, --help>
#   [options]       - Optional flags to modify behavior:
#       -r, --root    - Build from the root directory.
#       -d, --doc     - Build index.html && latex.pdf files using `doxygen`
#       -a, --all     - Clean all files in the build directory.
#       -t, --test    - Run tests after building the project.
#
# Example Usage:
#   ./run.sh -b             - Build the project
#   ./run.sh -r             - Run the project
#   ./run.sh -c             - Clean the project
#   ./run.sh -b -r          - Build and then run the project
#   ./run.sh -b -a          - Clean all files and build the project
# ====================================================================================

# Define color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No color

#* Define variables
APP_NAME="Racognition"
BUILD_DIR="bin/build"
CMAKE_LIST_DIR="conf"
CMAKE_GENERATOR="Ninja"
TEST_BUILD_DIR="$BUILD_DIR/test"
TEST_NAME="test"
BUILD_CMD="ninja"
ROOT_FLAG=false
ALL_FLAG=false
TEST_FLAG=false
DOC_FLAG=false
COMMAND=""

#* Parse options using a while loop
while [[ $# -gt 0 ]]; do
  case "$1" in
  build)
    COMMAND="build"
    shift
    ;;
  run)
    COMMAND="run"
    shift
    ;;
  clean)
    COMMAND="clean"
    shift
    ;;
  -h | --help)
    COMMAND=help
    shift
    ;;
  -t | --test)
    TEST_FLAG=true
    shift
    ;;
  -a | --all)
    ALL_FLAG=true
    shift
    ;;
  -r | --root)
    ROOT_FLAG=true
    shift
    ;;
  -d | --doc)
    DOC_FLAG=true
    shift
    ;;
  *)
    echo -e "${RED}Error:${NC} Invalid option: $1"
    exit 1
    ;;
  esac
done

#* Check if command is provided
if [ -z "$COMMAND" ]; then
  echo -e "${RED}Error:${NC} Command is required!"
  echo -e "${BLUE}Usage:${NC} ./run.sh <build|run|clean|setup|-h,--help> [-t,--test|-a,--all|-r,--root]"
  return 1
fi

#* Display help if --help or -h is passed
help() {
  echo -e "${BLUE}Usage:${NC} ./run.sh <build|run|clean|-h,--help> [-t,--test|-a,--all|-r,--root]"
  echo -e "${YELLOW}Commands:${NC}"
  echo -e "  build           Build the project"
  echo -e "  run             Run the project"
  echo -e "  clean           Clean the project"
  echo -e "  -h, --help      Display this help message and exit"
  echo -e "${YELLOW}Options:${NC}"
  echo -e "  -t, --test      Run tests after building the project"
  echo -e "  -a, --all       Clean all files in the build directory (Clear)"
  echo -e "  -r, --root      Build the project from the root (from CMakeLists.txt)"
  return 0
}

build() {
  #* Handle build with --root option
  if [ "$ROOT_FLAG" = true ]; then
    echo -e "${BLUE}Building project from root using CMake with generator '$CMAKE_GENERATOR'...${NC}"
    cmake -S "$CMAKE_LIST_DIR" -B "$BUILD_DIR" -G "$CMAKE_GENERATOR" || {
      echo -e "${RED}Error:${NC} CMake configuration failed."
      return 1
    }
    $BUILD_CMD -C "$BUILD_DIR" || {
      echo -e "${RED}Error:${NC} Build failed."
      return 1
    }
    echo -e "${GREEN}Build from root completed successfully.${NC}"
    return 0
  fi

  #* Handle build with --doc option
  # if [ "$DOC_FLAG" = true ]; then
  #   echo -e "${BLUE}Building documentation from Doxyfile using doxygen...${NC}"
  #   doxygen -S "$CMAKE_LIST_DIR" -B "$BUILD_DIR" -G "$CMAKE_GENERATOR" || {
  #     echo -e "${RED}Error:${NC} CMake configuration failed."
  #     return 1
  #   }
  #   $BUILD_CMD -C "$BUILD_DIR" || {
  #     echo -e "${RED}Error:${NC} Build failed."
  #     return 1
  #   }
  #   echo -e "${GREEN}Build from root completed successfully.${NC}"
  #   return 0
  # fi

  #* Handle regular build
  if [ -d "$BUILD_DIR" ]; then
    echo -e "${BLUE}Building project in '$BUILD_DIR'...${NC}"
    cd "$BUILD_DIR" || {
      echo -e "${RED}Error:${NC} Failed to enter build directory."
      return 1
    }
    $BUILD_CMD || {
      echo -e "${RED}Error:${NC} Build failed."
      return 1
    }
    echo -e "${GREEN}Build completed successfully.${NC}"
  else
    echo -e "${RED}Error:${NC} Build directory '$BUILD_DIR' not found."
    return 1
  fi
}

run() {
  #* Run the application if it exists
  if [ "$TEST_FLAG" = true ]; then
    #* Run tests after building
    echo -e "${BLUE}Running tests after building the project...${NC}"
    cd "$TEST_BUILD_DIR" || {
      echo -e "${RED}Error:${NC} Failed to enter test build directory."
      return 1
    }
    ./"$TEST_NAME" || {
      echo -e "${RED}Error:${NC} Application run failed."
      return 1
    }
    echo -e "${GREEN}Running tests...${NC}"
    ctest || {
      echo -e "${RED}Error:${NC} Test execution failed."
      return 1
    }
    echo -e "${GREEN}Tests completed successfully.${NC}"
    return 0
  fi

  #* Regular run (without tests)
  if [ -f "$BUILD_DIR/$APP_NAME" ]; then
    echo -e "${BLUE}Running application...${NC}"
    "$BUILD_DIR/$APP_NAME" || {
      echo -e "${RED}Error:${NC} Failed to run the application."
      return 1
    }
  else
    echo -e "${RED}Error:${NC} Application '$APP_NAME' not found in '$BUILD_DIR'."
    return 1
  fi
}

clean() {
  #* Handle clean with --all option
  if [ "$ALL_FLAG" = true ]; then
    if [ -d "$BUILD_DIR" ]; then
      echo -e "${BLUE}Removing all contents (including hidden files) inside '$BUILD_DIR'...${NC}"
      rm -rf "$BUILD_DIR"/* "$BUILD_DIR"/.* 2>/dev/null || {
        echo -e "${RED}Error:${NC} Failed to delete contents of build directory."
        return 1
      }
      echo -e "${GREEN}Clean operation completed successfully.${NC}"
    else
      echo -e "${RED}Error:${NC} Build directory '$BUILD_DIR' not found."
      return 1
    fi
    return 0
  fi

  #* Clean the build directory (without --all)
  if [ -d "$BUILD_DIR" ]; then
    echo -e "${BLUE}Cleaning build directory...${NC}"
    cd "$BUILD_DIR" || {
      echo -e "${RED}Error:${NC} Failed to enter build directory."
      return 1
    }
    $BUILD_CMD clean || {
      echo -e "${RED}Error:${NC} Clean failed."
      return 1
    }
    echo -e "${GREEN}Clean completed successfully.${NC}"
  else
    echo -e "${RED}Error:${NC} Build directory '$BUILD_DIR' not found."
    return 1
  fi
}

#* Process commands
case "$COMMAND" in
build)
  build
  ;;
run)
  run
  ;;
clean)
  clean
  ;;
help)
  help
  return 0
  ;;
*)
  echo -e "${RED}Error:${NC} Unknown command '$COMMAND'!"
  echo -e "${BLUE}Usage:${NC} ./run.sh <build|run|clean|-h,--help> [-t,--test|-a,--all|-r,--root]"
  ;;
esac
