import torch
import argparse
import torch.nn as nn
import random
import numpy as np
import os

from ingenic_magik_trainingkit.QuantizationTrainingPlugin.python.fqat.quantization.quantization_fqat import MagikQuantizer
from ingenic_magik_trainingkit.QuantizationTrainingPlugin.python import quantize_ops
from ingenic_magik_trainingkit.QuantizationTrainingPlugin.python.calibration_data_reader import seed_all
from utils.torch_utils import EarlyStopping, ModelEMA, de_parallel, select_device, torch_distributed_zero_first
from ingenic_magik_trainingkit.QuantizationTrainingPlugin.python.data_reader import Yolov5DataReader
from ingenic_magik_trainingkit.QuantizationTrainingPlugin.python.fqat.quantization.utils import magik_export_onnx

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

seed_all(1000)

path = "calibrate.txt"#"./data/coco/person_1.5w.txt"#
cal_data_reader = Yolov5DataReader(path=path, nbatchs=4, batch_size=4, img_size=(640, 640), mean=0, var=255) # mean and variance


def get_magik_quantizer(bit=4):
    # get quantize_parameter
    quantize_parameter = {"weight_bitwidth" : bit,
                          "output_bitwidth" : bit,
                          "target_device" : "T33", # platform
                          "feature_quantize_method" : "DAQS",
                          "feature_calibration_method" : "MSE",
                          "weight_quantize_method" : "DAQS",
                          "weight_calibration_method" : "ADMM",
                          "feature_calibration_loop_size" : 80, # iteration times of feature calibration
                          "weight_calibration_loop_size" : 200, # iteration times of weight calibration
                          "fixpoint" : True, # fixpoint can be aligned with board end data
                          "set_ahead_layers_8bit" : 2, # set the first n layers to 8 bit
                          "set_hinder_layers_8bit" : 2, # set the last n layers to 8 bit
                          "input_mean" : [0, 0, 0], # mean
                          "input_var" : [255, 255, 255], # variance
                          "quantize_model_input" : False, # quantize input of the network or not
                         }

    magik_quantizer = MagikQuantizer(quantize_parameter,cal_data_reader)

    return magik_quantizer

@torch.no_grad()
def get_and_save_quant_model(magik_quantizer, weight, cfg, onnx):
    from models.yolo import Model
    net = Model(cfg, ch=3, nc=1).cuda()
    chkpt = torch.load(args.weights)
    net.load_state_dict(chkpt['model'].float().state_dict(), strict=True)
    net.to(device)
    net.eval()

    net = magik_quantizer.quant(net)

    chkpt['ema'] = net
    if onnx:
        dummy_input = torch.randn(1, 3, 640, 640).to(device)
        dst_path = "checkpoint/8W8F/weights/best-tmp.onnx"
        magik_export_onnx(net, dummy_input, dst_path)
        print("---convert---over----")
        exit()
    torch.save(chkpt, "checkpoint/8W8F/weights/best-tmp.pt")
    print("pqt model over----!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Quant')
    parser.add_argument("--weights", type=str, default="runs/train/yolov5s-person-float/weights/best.pt", help="path to model")
    parser.add_argument("--cfg", type=str, default="./yolov5s-person.yaml", help="path to model")
    parser.add_argument('--bit', type=int, default=8, help='You should set bit to 32 or 8 or 4')
    parser.add_argument('--onnx', action='store_true', help='use OpenCV DNN for ONNX inference')
    args = parser.parse_args()
    magik_quantizer = get_magik_quantizer(args.bit)
    get_and_save_quant_model(magik_quantizer, args.weights, args.cfg, args.onnx)
