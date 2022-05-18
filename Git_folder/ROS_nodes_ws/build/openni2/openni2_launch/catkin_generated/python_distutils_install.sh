#!/bin/sh

if [ -n "$DESTDIR" ] ; then
    case $DESTDIR in
        /*) # ok
            ;;
        *)
            /bin/echo "DESTDIR argument must be absolute... "
            /bin/echo "otherwise python's distutils will bork things."
            exit 1
    esac
fi

echo_and_run() { echo "+ $@" ; "$@" ; }

echo_and_run cd "/home/alex/Alex_documents/RGCNN_git/Git_folder/ROS_nodes_ws/src/openni2/openni2_launch"

# ensure that Python install destination exists
echo_and_run mkdir -p "$DESTDIR/home/alex/Alex_documents/RGCNN_git/Git_folder/ROS_nodes_ws/install/lib/python2.7/dist-packages"

# Note that PYTHONPATH is pulled from the environment to support installing
# into one location when some dependencies were installed in another
# location, #123.
echo_and_run /usr/bin/env \
    PYTHONPATH="/home/alex/Alex_documents/RGCNN_git/Git_folder/ROS_nodes_ws/install/lib/python2.7/dist-packages:/home/alex/Alex_documents/RGCNN_git/Git_folder/ROS_nodes_ws/build/lib/python2.7/dist-packages:$PYTHONPATH" \
    CATKIN_BINARY_DIR="/home/alex/Alex_documents/RGCNN_git/Git_folder/ROS_nodes_ws/build" \
    "/usr/bin/python2" \
    "/home/alex/Alex_documents/RGCNN_git/Git_folder/ROS_nodes_ws/src/openni2/openni2_launch/setup.py" \
     \
    build --build-base "/home/alex/Alex_documents/RGCNN_git/Git_folder/ROS_nodes_ws/build/openni2/openni2_launch" \
    install \
    --root="${DESTDIR-/}" \
    --install-layout=deb --prefix="/home/alex/Alex_documents/RGCNN_git/Git_folder/ROS_nodes_ws/install" --install-scripts="/home/alex/Alex_documents/RGCNN_git/Git_folder/ROS_nodes_ws/install/bin"
