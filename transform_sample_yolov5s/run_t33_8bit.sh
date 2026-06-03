cd ../
sed -ri 's/(target_device=")[^"]*/\1T33/' models/yolo.py models/common.py
sed -ri 's/(target_device" : ")[^"]*/\1T33/' magik_quantize.py

# 8bit .pt -> .onnx
python detect.py \
  --source data/images/bus.jpg \
  --float_model checkpoint/32W32F/weights/best.pt \
  --weights checkpoint/8W8F/weights/best-tmp.pt \
  --imgs 640 \
  --device 0 \
  --bit 8 \
  --onnx

cd -

# 8bit .onnx -> .mgk
../magik-transform-tools \
  --inputpath ../checkpoint/8W8F/weights/best-tmp.onnx \
  --outputpath ./yolov5s_t33_magik_fqat.mk.h \
  --config cfg/magik_t33.cfg

cp -r compile_result/glibc/ uranus_sample_yolov5s-person-fqat/model_t33/
cp -r compile_result/uclibc/ uranus_sample_yolov5s-person-fqat/model_t33/
rm -r compile_result magik_transform_event_*.log
