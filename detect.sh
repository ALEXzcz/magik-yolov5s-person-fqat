#for detect single img
# float
# python detect.py --source data/images/bus.jpg --float_model checkpoint/32W32F/weights/best.pt --weights ./checkpoint/32W32F/weights/best.pt --imgs 640 --device 0 --view-img --model_only

# 8bit
# python detect.py --source data/images/bus.jpg --float_model checkpoint/32W32F/weights/best.pt --weights ./checkpoint/8W8F/weights/best.pt --imgs 640 --device 0 --view-img --bit 8

# 4bit
python detect.py --source data/images/bus.jpg --float_model checkpoint/32W32F/weights/best.pt --weights ./checkpoint/4W4F/weights/best.pt --imgs 640 --device 0 --view-img --bit 4

#for onnx:
# 8bit
# python detect.py --source data/images/bus.jpg --float_model checkpoint/32W32F/weights/best.pt --weights ./checkpoint/8W8F/weights/best.pt --imgs 640 --device 0 --view-img --onnx --bit 8

# 4bit
# python detect.py --source data/images/bus.jpg --float_model checkpoint/32W32F/weights/best.pt --weights ./checkpoint/4W4F/weights/best.pt --imgs 640 --device 0 --view-img --onnx --bit 4
