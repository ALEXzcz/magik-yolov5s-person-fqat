##### SOC T32 #####
export LD_LIBRARY_PATH=$(pwd)/../InferenceKit/uranus/tnna/540/lib/uclibc:${LD_LIBRARY_PATH}

# for rgba model
./load_mgk_uclibc_release model_t33/uclibc/yolov5s_t33_magik_fqat.mgk test_image/bus_640x640_bgr.jpg
# ./load_mgk_uclibc_release model_t33/uclibc/yolov5s_t33_magik_fqat.mgk test_image/bus_640x640_nv12.yuv 640 640
