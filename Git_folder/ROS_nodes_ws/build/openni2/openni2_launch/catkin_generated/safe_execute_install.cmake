execute_process(COMMAND "/home/alex/Alex_documents/RGCNN_git/Git_folder/ROS_nodes_ws/build/openni2/openni2_launch/catkin_generated/python_distutils_install.sh" RESULT_VARIABLE res)

if(NOT res EQUAL 0)
  message(FATAL_ERROR "execute_process(/home/alex/Alex_documents/RGCNN_git/Git_folder/ROS_nodes_ws/build/openni2/openni2_launch/catkin_generated/python_distutils_install.sh) returned error code ")
endif()
