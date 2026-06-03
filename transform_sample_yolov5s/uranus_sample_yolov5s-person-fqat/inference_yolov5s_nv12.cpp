/*
 *        (C) COPYRIGHT Ingenic Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : inference_yolov5s_nv12.cpp
 * Authors    : ljhong, cywang
 * Create Time: 2024-11-21:16:51:14
 * Description:
 * 
 */

#define STB_IMAGE_IMPLEMENTATION
#include "./stb/stb_image.h"
#include "./stb/drawing.hpp"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "./stb/stb_image_write.h"
static const uint8_t color[3] = {0xff, 0, 0};

#include "ingenic_mnni.h"
#include "common/common_utils.h"
#include "utils/imgproc.h"
#include "utils/postproc.h"

#include <cmath>
#include <errno.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <cstring>
#include <string.h>
#include <vector>
#include <algorithm>
#include <chrono>

#define IS_ALIGN_64(x) (((size_t)x) & 0x3F)

const int input_w = 640;
const int input_h = 640;
float person_threshold = 0.5;
float nms_threshold = 0.6;
int classes = 1;
vector<float> strides = {8.0, 16.0, 32.0};
int box_num = 3;
std::vector<float> anchor = {10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326};

using namespace std;
using namespace magik::uranus;

typedef unsigned char uint8_t;
#define RET_CHECKER(ret, msg)                                           \
    do {                                                                \
        if (!(ret)) {                                                   \
            fprintf(stderr, "%s [%s %d]\n", msg, __func__, __LINE__);   \
            return -1;                                                  \
        }                                                               \
    } while(0)

typedef struct {
    uint8_t *data;
    int width;
    int height;
}FrameInfo;

void *read_nv12(FrameInfo &in, const std::string &img_path, int img_w, int img_h) {
    FILE *fp = NULL;
    int nv12_size = img_w * img_h * 1.5;
    uint8_t *nv12_data = (uint8_t *)malloc(nv12_size);
    fp = fopen(img_path.c_str(), "rb");
    if (fp) {
        int ret = fread(nv12_data, 1, nv12_size, fp);
        printf("\033[32mret %d\033[0m\n", ret);
        fclose(fp);
    } else {
        printf("open nv12 file failed\n");
    }
    in.data = nv12_data;
    in.width = img_w;
    in.height = img_h;
    return nullptr;
}

void generateBBox(std::vector<magik::uranus::Tensor> out_res, std::vector<magik::uranus::ObjBbox_t>& candidate_boxes, int img_w, int img_h) {   
    DetectorType_t detector_type = DetectorType_t::YOLOV5;
    magik::uranus::generate_box(out_res, strides, anchor, candidate_boxes, img_w, img_h, classes, box_num, person_threshold, detector_type);
}

void manyclass_nms(std::vector<magik::uranus::ObjBbox_t> &input, std::vector<magik::uranus::ObjBbox_t> &output, int classnums, int type) {
    vector<magik::uranus::ObjBbox_t> output_tmp;
    output_tmp.clear();
    output.clear();
    int box_num = input.size();
    std::vector<int> merged(box_num, 0);
    std::vector<magik::uranus::ObjBbox_t> classbuf;
    for (int clsid = 0; clsid < classnums; clsid++) {
        classbuf.clear();
        for (int i = 0; i < box_num; i++) {
            if (merged[i])
                continue;
            if(clsid!=input[i].class_id)
                continue;
            classbuf.push_back(input[i]);
            merged[i] = 1;
        }
        magik::uranus::nms(classbuf, output_tmp, nms_threshold, magik::uranus::NmsType::HARD_NMS);
        for(int i = 0; i < (int)output_tmp.size(); i++){
            output_tmp[i].class_id = clsid;
            output.push_back(output_tmp[i]);
        }
        output_tmp.clear();
    }
}

void scale_boxes(int ori_w, int ori_h, int new_w, int new_h, std::vector<magik::uranus::ObjBbox_t> &out_boxes) {
    float scale;
    float pad_x, pad_y;
    scale = std::min((float(new_w) / ori_w), (float(new_h) / ori_h));
    pad_x = (new_w - ori_w * scale) / 2;
    pad_y = (new_h - ori_h * scale) / 2;
    for (unsigned int i = 0; i < out_boxes.size(); i++)
    {
        out_boxes[i].box.x0 = round(std::max((out_boxes[i].box.x0 - pad_x) / scale, float(0)));
        out_boxes[i].box.x1 = round(std::min((out_boxes[i].box.x1 - pad_x) / scale, float(ori_w - 1)));
        out_boxes[i].box.y0 = round(std::max((out_boxes[i].box.y0 - pad_y) / scale, float(0)));
        out_boxes[i].box.y1 = round(std::min((out_boxes[i].box.y1 - pad_y) / scale, float(ori_h - 1)));
    }
}

void run_model(magik::uranus::Network *model, FrameInfo input_info, std::vector<magik::uranus::Tensor> &outputs) {
    auto input_names = model->get_input_names();
    for(auto n: input_names) {
        std::cout<<"input_names: "<< n <<std::endl;
        auto input = model->get_input(n);
        uint8_t *indata = (uint8_t*)(input.vdata<uint8_t>());
        memcpy((void *)indata, (void *)input_info.data, input_info.width * input_info.height * 1.5 * sizeof(uint8_t));
    }

    // net forword
    auto start = std::chrono::high_resolution_clock::now();

    int RUN_CNT = 1;
    for (int i = 0; i < RUN_CNT; i++) {
        model->run();
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Time taken: " << duration << " ms" << std::endl;

    // get outputs
    auto output_names = model->get_output_names();
    for(auto n: output_names) {
        std::cout<<"output_names: "<< n <<std::endl;
        auto output = model->get_output(n);
        outputs.push_back(output);
    }
    return;
}

int run_detect(std::string model_path, std::string img_path, int img_w, int img_h) {
    /******************load model***********************/
    Network **derived_models;
    int model_num = 0;
    printf("load model: %s \n", model_path.c_str());
    if (magik_load_model(derived_models, model_num, model_path.c_str()))
    {
        printf("create model failed.\n");
        exit(1);
    }

    /** Show magik model version. **/
    // show_magik_version(model_path.c_str());

    auto model = derived_models[0];
    /******************prepare data***********************/
    printf("load image: %s \n", img_path.c_str());
    FrameInfo input_info;
    printf("read_nv12=========\n");
    read_nv12(input_info, img_path.c_str(), img_w, img_h);
    RET_CHECKER(input_info.data != nullptr, "Read image failed!!!!");

    /******************model forward***********************/

    /** Gets the size of the memory space required for the network forward **/
    int ret = model->get_forward_memory_size();
    std::cout << "model forward memory size: " << ret << " = " << ret / 1024 << "KB" << " = " << ret / 1024 / 1024 << "MB" << std::endl;

    std::vector<magik::uranus::Tensor> outputs;
    run_model(model, input_info, outputs);

    /******************postprocess***********************/
    std::vector<magik::uranus::ObjBbox_t> temp_boxes;
    vector<magik::uranus::ObjectBbox_t> output_boxes;
    generateBBox(outputs, temp_boxes, input_w, input_h);
    if (classes == 1) {
        magik::uranus::nms(temp_boxes, output_boxes, nms_threshold, magik::uranus::NmsType_t::HARD_NMS);
    } else {
        manyclass_nms(temp_boxes, output_boxes, classes, 0);
    }
    scale_boxes(input_info.width, input_info.height, input_w, input_h, output_boxes);
    int num_boxes = output_boxes.size();
    std::cout << "num_boxes: " << num_boxes << std::endl;
    for (int i = 0; i < int(output_boxes.size()); i++) {
        auto output_box = output_boxes[i];
        printf("box:   ");
        printf("x0: %d ", (int)output_box.box.x0);
        printf("y0: %d ", (int)output_box.box.y0);
        printf("x1: %d ", (int)output_box.box.x1);
        printf("y1: %d ", (int)output_box.box.y1);
        printf("score: %.2f ", output_box.score);
        printf("class: %d ", output_box.class_id);
        printf("\n");
    }
    free(input_info.data);
    /******************destroy model***********************/
    magik_destroy_model(model);
    printf("magik_destroy_model over\n");

	std::cout << "^_^ Run successfully ^_^" << std::endl;
    return 0;
}

int main(int argc, char **argv) {
    std::string model_path = argv[1];
    std::string img_path = argv[2];
    int ori_img_w = atoi(argv[3]);
    int ori_img_h = atoi(argv[4]);

    if (argc != 5)
    {
        printf("%s model_path nv12_path w h\n", argv[0]);
        exit(0);
    }
    run_detect(model_path,img_path,ori_img_w,ori_img_h);
    return 0;
}
