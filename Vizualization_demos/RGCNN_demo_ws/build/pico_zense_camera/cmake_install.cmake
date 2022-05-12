# Install script for directory: /home/alex/Alex_documents/RGCNN_git/Vizualization_demos/RGCNN_demo_ws/src/pico_zense_camera

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/alex/Alex_documents/RGCNN_git/Vizualization_demos/RGCNN_demo_ws/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/home/alex/Alex_documents/RGCNN_git/Vizualization_demos/RGCNN_demo_ws/build/pico_zense_camera/catkin_generated/safe_execute_install.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/pico_zense_camera/cmake" TYPE FILE FILES "/home/alex/Alex_documents/RGCNN_git/Vizualization_demos/RGCNN_demo_ws/build/pico_zense_camera/catkin_generated/installspace/pico_zense_camera-msg-paths.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/roseus/ros" TYPE DIRECTORY FILES "/home/alex/Alex_documents/RGCNN_git/Vizualization_demos/RGCNN_demo_ws/devel/share/roseus/ros/pico_zense_camera")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  execute_process(COMMAND "/usr/bin/python2" -m compileall "/home/alex/Alex_documents/RGCNN_git/Vizualization_demos/RGCNN_demo_ws/devel/lib/python2.7/dist-packages/pico_zense_camera")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/python2.7/dist-packages" TYPE DIRECTORY FILES "/home/alex/Alex_documents/RGCNN_git/Vizualization_demos/RGCNN_demo_ws/devel/lib/python2.7/dist-packages/pico_zense_camera" REGEX "/\\_\\_init\\_\\_\\.py$" EXCLUDE REGEX "/\\_\\_init\\_\\_\\.pyc$" EXCLUDE)
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/python2.7/dist-packages" TYPE DIRECTORY FILES "/home/alex/Alex_documents/RGCNN_git/Vizualization_demos/RGCNN_demo_ws/devel/lib/python2.7/dist-packages/pico_zense_camera" FILES_MATCHING REGEX "/home/alex/Alex_documents/RGCNN_git/Vizualization_demos/RGCNN_demo_ws/devel/lib/python2.7/dist-packages/pico_zense_camera/.+/__init__.pyc?$")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/pico_zense_camera" TYPE FILE FILES "/home/alex/Alex_documents/RGCNN_git/Vizualization_demos/RGCNN_demo_ws/devel/include/pico_zense_camera/pico_zense_dcam710Config.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  execute_process(COMMAND "/usr/bin/python2" -m compileall "/home/alex/Alex_documents/RGCNN_git/Vizualization_demos/RGCNN_demo_ws/devel/lib/python2.7/dist-packages/pico_zense_camera/cfg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/python2.7/dist-packages/pico_zense_camera" TYPE DIRECTORY FILES "/home/alex/Alex_documents/RGCNN_git/Vizualization_demos/RGCNN_demo_ws/devel/lib/python2.7/dist-packages/pico_zense_camera/cfg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES "/home/alex/Alex_documents/RGCNN_git/Vizualization_demos/RGCNN_demo_ws/build/pico_zense_camera/catkin_generated/installspace/pico_zense_camera.pc")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/pico_zense_camera/cmake" TYPE FILE FILES "/home/alex/Alex_documents/RGCNN_git/Vizualization_demos/RGCNN_demo_ws/build/pico_zense_camera/catkin_generated/installspace/pico_zense_camera-msg-extras.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/pico_zense_camera/cmake" TYPE FILE FILES
    "/home/alex/Alex_documents/RGCNN_git/Vizualization_demos/RGCNN_demo_ws/build/pico_zense_camera/catkin_generated/installspace/pico_zense_cameraConfig.cmake"
    "/home/alex/Alex_documents/RGCNN_git/Vizualization_demos/RGCNN_demo_ws/build/pico_zense_camera/catkin_generated/installspace/pico_zense_cameraConfig-version.cmake"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/pico_zense_camera" TYPE FILE FILES "/home/alex/Alex_documents/RGCNN_git/Vizualization_demos/RGCNN_demo_ws/src/pico_zense_camera/package.xml")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/pico_zense_camera" TYPE DIRECTORY FILES "/home/alex/Alex_documents/RGCNN_git/Vizualization_demos/RGCNN_demo_ws/src/pico_zense_camera/launch")
endif()

