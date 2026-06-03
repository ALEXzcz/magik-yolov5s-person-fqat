# yolov5s_person
origin code , see https://github.com/ultralytics/yolov5
commit: c4cb7c684ad6baf4081eb59ee1c27aa97bed4ebe
The network structure has not been modified, and individual operators may be added or modified to adapt to our plugins and subsequent transformations. Please refer to the specific code for details.

In models/experimental.py In line 98, we cancel the integration of Conv and Bn in the model. Because our plug-in will integrate Conv and Bn, we will not repeat the operation.

# testing environment
- python == 3.8
- torch == 1.8
- cuda == 10.2
- MagikTrainingKit == 2.1.0(commit:88e80c034e0162edde9d09494fe74d46c0283434)

# the code structure
```console
yolov5s-person/
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

# Dependencies
see requirements.txt

## Install dependent environment
```console
$ pip install -r requirements.txt
```

## Install MagikTransformer and  TrainingKit
```console
$git clone http://magik:crMbZ6cyXjzqbm405JDx0vu27GAl8Y+2ZCQ4cciWaw@218.106.92.37:8083/gerrit/magik/magik-toolkit
```
And you will see some directory as the folllowing, to get these whl files.
```console
magik-toolkit
├── TrainingKit
│   └── pytorch
│       ├── magik_whl
│       │   ├── Magik_py38_torch1.8_cuda10.2
│       │   │       ├── magik_trainingkit_torch_180-1.1.1-py3-none-any.whl
├── TransformKit
│   ├── magik_transformer-2.0.5-py3-none-any.whl
```
Attention: you should download magik_trainingkit_torch_****.whl for matching your python, pytorch,cuda version,for example:
```console
$ cd magik-toolkit/TrainingKit/pytorch/magik_whl/Magik_py36_torch1.8_cuda10.2/
$ pip install magik_trainingkit_torch_180-1.1.1-py3-none-any.whl
$ cd magik-toolkit/TransformKit/
$ pip install magik_transformer-2.0.5-py3-none-any.whl
```

# Float Model
## Train
float model train(see train.sh):
lr: 0.01(data/hyp.scratch-float.yaml)
optimizer: SGD

## Eval
test precision（see test.sh）
```console
python val.py --data data/coco-person.yaml --float_model checkpoint/32W32F/weights/best.pt --weights ./checkpoint/32W32F/weights/best.pt --imgs 640 --device 0 --batch-size 100
```

# 8bit Model

## Train
8bit model coming from float directly(see magik.sh)

```console
python magik_quantize.py --cfg models/yolov5s.yaml --weights ./checkpoint/32W32F/weights/best.pt --bit 8
```

And the .pth will be saved in checkpoint/8W8F/weights/best-tmp.pt

## Eval
(see test.sh):
```console
python val.py --data data/coco-person.yaml --float_model checkpoint/32W32F/weights/best.pt --weights ./checkpoint/8W8F/weights/best.pt --imgs 640 --device 0 --batch-size 50 --bit 8
```

# 4bit Model
## Train
The direct conversion accuracy of the 4-bit model is not high, and it needs to be pretrained with the original float model (see train. sh):
lr: 0.001(data/hyp.scratch.yaml)
optimizer: Adam

## Eval
```console
python val.py --data data/coco-person.yaml --float_model checkpoint/32W32F/weights/best.pt --weights ./checkpoint/4W4F/weights/best.pt --imgs 640 --device 0 --batch-size 50 --bit 4
```

# Detect
Single image testing facilitates data verification
```console
python detect.py --source data/images/bus.jpg --float_model checkpoint/32W32F/weights/best.pt --weights ./checkpoint/4W4F/weights/best.pt --imgs 640 --device 0 --view-img --bit 4
```

# Export Onnx
```console
python detect.py --source data/images/bus.jpg --float_model checkpoint/32W32F/weights/best.pt --weights ./checkpoint/4W4F/weights/best.pt --imgs 640 --device 0 --view-img --onnx --bit 4
```

## Common errors when export onnx
- Q: "step!=1 is currently not supported" appears when converting from torch to onnx
- A: Problems encountered when converting yolov5 slices. It works in versions 1.6 and earlier, but becomes an error after 1.7. Just comment out the corresponding line if + raise RuntimeError in torch/onnx/symbolic_opset9.py.
- Q: Error when converting from torch to onnx "RuntimeError: input_shape_value==reshape_value || input_shape_value==1 || reshape_value == 1 INTERNAL ASSERT FAILED ..."
- A: Error after version 1.9. The last parameter of _export() in line 635 of torch/onnx/utils.py is changed from onnx_shape_inference=True to onnx_shape_inference=False;
torch1.10 is on line 677. Find this function in different versions and modify it accordingly.

# Transform model
```console
python ./transform_sample/transform.py --model_file ./checkpoint/4W4F/weights/best.onnx --output_file yolov5s_person_4bit.mk.h --config_file ./transform_sample/cfg/magik_t40.cfg
```

# Precision
- python3.8 torch1.8 cuda10.2
- MagikTrainingKit == 3.0.0(commit:29d75d67a64b2cc5d4102f8113d2acee01385c0f)

|Model|Class|Images|Labels|P|R|mAP@.5|mAP@.5:.95|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|float    |all       |5000      |11004      |0.762      |0.612      |0.690      |0.413|
|8bit     |all       |5000      |11004      |0.737      |0.63      |0.691      |0.411|
|4bit     |all       |5000      |11004      |0.778      |0.598      |0.687      |0.411|

# Run model on The plate end


