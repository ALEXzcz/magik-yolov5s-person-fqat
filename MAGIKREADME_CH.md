# 原代码地址:
https://github.com/ultralytics/yolov5
commit:c4cb7c684ad6baf4081eb59ee1c27aa97bed4ebe
网络结构未改动，可能添加或修正个别算子以适应我们的插件及后续转换，详见具体代码。

# 测试环境
- python == 3.8
- torch == 1.8
- cuda == 10.2
- MagikTrainingKit == 3.0.0(commit:29d75d67a64b2cc5d4102f8113d2acee01385c0f)

## 环境依赖
```console
$ pip install -r requirements.txt
```

## MagikTrainingKit
```console
$ git clone http://magik:crMbZ6cyXjzqbm405JDx0vu27GAl+2ZCQ4cciWaw@218.106.92.37:8083/gerrit/magik/magik-toolkit
$ cd magik-toolkit/TrainingKit/pytorch/magik_whl/Magik_py38_torch1.8_cuda10.2/
$ pip install magik_trainingkit_torch_180-2.1.0-py3-none-any.whl
```

## MagikTransformKit
```console
$ git clone http://magik:crMbZ6cyXjzqbm405JDx0vu27GAl+2ZCQ4cciWaw@218.106.92.37:8083/gerrit/magik/magik-toolkit
$ cd magik-toolkit/TransformKit
$ pip install magik_transformer-2.0.6-py3-none-any.whl
```

# 代码结构
```
console
yolov5s-person
├── calibrate.txt
├── checkpoint
├── CONTRIBUTING.md
├── data
├── detect.py
├── detect.sh
├── export.py
├── generate_datasets
├── hubconf.py
├── LICENSE
├── magik_quantize.py
├── MAGIKREADME_CH.md
├── MAGIKREADME_EN.md
├── magik.sh
├── models
├── __pycache__
├── README.md
├── requirements.txt
├── runs
├── setup.cfg
├── test.sh
├── train.py
├── train.sh
├── transform_sample
├── tutorial.ipynb
├── utils
└── val.py
```

# Float

## 训练(参见train.sh):
```console
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=info
GPUS=4
python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=60051 train.py \
     --data data/coco-person.yaml\
     --cfg models/yolov5s.yaml \
     --weights '' \
     --batch-size 64 \
     --hyp data/hyp.scratch-float.yaml \
     --project ./runs/train/yolov5s-person-float \
     --epochs 300 \
     --device 0,1,2,3 \
     --bit 32
```

## 测试(参见test.sh):
```console
python val.py --data data/coco-person.yaml --float_model checkpoint/32W32F/weights/best.pt --weights ./checkpoint/32W32F/weights/best.pt --imgs 640 --device 0 --batch-size 100
```

# 8bit模型

## 直接转换(参见magik.sh):
```console
python magik_quantize.py --cfg models/yolov5s.yaml --weights ./checkpoint/32W32F/weights/best.pt --bit 8
```

## 测试(参见test.sh):
```console
python val.py --data data/coco-person.yaml --float_model checkpoint/32W32F/weights/best.pt --weights ./checkpoint/8W8F/weights/best.pt --imgs 640 --device 0 --batch-size 100 --bit 8
```

# 4bit模型

## 训练，4bit模型直接转换精度不高，需要以float模型为预训练模型进行训练(参见train.sh):
```console
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=info
GPUS=4
python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=60051 train.py \
    --data data/coco-person.yaml\
    --cfg models/yolov5s.yaml \
    --weights 'checkpoint/32W32F/weights/best.pt' \
    --batch-size 64 \
    --hyp data/hyp.scratch.yaml \
    --project ./runs/train/yolov5s-person-4bit \
    --epochs 300 \
    --device 0,1,2,3 \
    --optimizer Adam \
    --bit 4
```

## 测试(参见test.sh):
```console
python val.py --data data/coco-person.yaml --float_model checkpoint/32W32F/weights/best.pt --weights ./checkpoint/4W4F/weights/best.pt --imgs 640 --device 0 --batch-size 100 --bit 4
```

# 测试单张图片,单张图片的测试方便核对数据(参见detect.sh)
```console
python detect.py --source data/images/bus.jpg --float_model checkpoint/32W32F/weights/best.pt --weights ./checkpoint/4W4F/weights/best.pt --imgs 640 --device 0 --view-img --bit 4
```

# 输出onnx(参见detect.sh)
```console
python detect.py --source data/images/bus.jpg --float_model checkpoint/32W32F/weights/best.pt --weights ./checkpoint/4W4F/weights/best.pt --imgs 640 --device 0 --view-img --onnx --bit 4
```

## 输出onnx常见报错
- Q：torch转onnx出现“step!=1 is currently not supported”
- A：yolov5 slice转换遇到的问题, 1.6及之前版本可以,1.7以后变成error了，torch/onnx/symbolic_opset9.py对应的行if + raise RuntimeError注掉即可。
- Q：torch转onnx出错“RuntimeError:input_shape_value==reshape_value || input_shape_value==1 || reshape_value == 1 INTERNAL ASSERT FAILED ……” 
- A：1.9版本之后的错，torch/onnx/utils.py中635行_export()最后一个参数的onnx_shape_inference=True改为 onnx_shape_inference=False；
torch1.10是在677行。不同版本找这个函数相应修改即可。

# 模型转换(参见run_t40.sh)
```console
CUDA_VISIBLE_DEVICES=0 python transform.py --model_file ../checkpoint/4W4F/best_bit4.onnx --output_file ./venus_sample_yolov8/yolov8s_t40_magik_4bit.mk.h --config_file cfg/magik_t40.cfg
```

# 模型精度
|Model|Class|Images|Labels|P|R|mAP@.5|mAP@.5:.95|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|Float|all|5000|11004|0.762|0.612|0.690|0.413|
|8bit |all|5000|11004|0.737|0.63|0.691|0.411|
|4bit |all|5000|11004|0.778|0.598|0.687|0.411|

# 注：在models/experimental.py中第98行，我们取消了模型中的Conv与Bn的融合，因为我们的插件中会将Conv和Bn进行融合，所以这里不再重复操作。