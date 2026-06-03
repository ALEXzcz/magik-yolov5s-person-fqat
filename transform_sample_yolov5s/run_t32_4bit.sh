cd ../
sed -ri 's/(target_device=")[^"]*/\1T32/' models/yolo.py models/common.py
sed -ri 's/(target_device" : ")[^"]*/\1T32/' magik_quantize.py

# 4bit .pt -> .onnx
python detect.py \
  --source data/images/bus.jpg \
  --float_model checkpoint/32W32F/weights/best.pt \
  --weights checkpoint/4W4F/weights/best-tmp.pt \
  --imgs 640 \
  --device 0 \
  --bit 4 \
  --onnx

cd -

# 4bit .onnx -> .mgk
../magik-transform-tools \
  --inputpath ../checkpoint/4W4F/weights/best-tmp.onnx \
  --outputpath ./yolov5s_t32_magik_fqat.mk.h \
  --config cfg/magik_t32.cfg

cp -r compile_result/glibc/ uranus_sample_yolov5s-person-fqat/model_t32/
cp -r compile_result/uclibc/ uranus_sample_yolov5s-person-fqat/model_t32/
rm -r compile_result magik_transform_event_*.log
