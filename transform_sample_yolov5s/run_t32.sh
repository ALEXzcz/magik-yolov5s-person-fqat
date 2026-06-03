cd ../
sed -ri 's/(target_device=")[^"]*/\1T32/' models/yolo.py models/common.py
sed -ri 's/(target_device" : ")[^"]*/\1T32/' magik_quantize.py
python detect.py --source data/images/bus.jpg --weights ./checkpoint/4W4F/weights/best.pt --imgs 640 --device 0 --view-img --onnx --bit 4
cd -

../../../../TransformKit/magik-transform-tools \
	--inputpath ../checkpoint/4W4F/weights/best.onnx \
	--outputpath ./yolov5s_t32_magik_fqat.mk.h \
	--config cfg/magik_t32.cfg

cp -r compile_result/glibc/ uranus_sample_yolov5s-person-fqat/model_t32/
cp -r compile_result/uclibc/ uranus_sample_yolov5s-person-fqat/model_t32/
rm -r compile_result magik_transform_event_*.log
