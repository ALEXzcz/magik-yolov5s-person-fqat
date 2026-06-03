/*
 *        (C) COPYRIGHT Ingenic Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : inference_yolov5s.cpp
 * Authors    : ljhong, cywang
 * Create Time: 2024-11-21:15:57:21
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
int ori_img_w;
int ori_img_h;
float person_threshold = 0.5;
float nms_threshold = 0.6;
int classes = 1;
vector<float> strides = {8.0, 16.0, 32.0};
int box_num = 3;
std::vector<float> anchor = {10, 13, 16, 30, 33, 23, 
                             30, 61, 62, 45, 59, 119, 
                             116, 90, 156, 198, 373, 326};

using namespace std;
using namespace magik::uranus;

#ifdef URANUS_NV12
magik::uranus::ChannelLayout DATA_FORMAT = magik::uranus::ChannelLayout::NV12;
#else
magik::uranus::ChannelLayout DATA_FORMAT = magik::uranus::ChannelLayout::RGBA;
#endif

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

static int fill_dim(uint8_t *data, int h, int w) {
    for (int i = 0; i < h*w; i++) {
        data[i*4+3] = 0;
    }
    return 0;
}

int rgb2bgr(uint8_t *data, int data_cnt){
    uint8_t* p = data;
    int pixel_c = data_cnt / 4;
    for (int i=0; i < pixel_c; ++i) {
        uint8_t t = p[0];
        p[0] = p[2];
        p[2] = t;
        p += 4;
    }
    return 0;
}

void read_img(FrameInfo &in, Img &img, const std::string &img_path) {
    int comp = 0;
    int w, h, c;
    unsigned char *imagedata = stbi_load(img_path.c_str(), &ori_img_w, &ori_img_h, &comp, 3); // image format is rgba
    in.data = stbi_load(img_path.c_str(), &w, &h, &c, 4);
    in.width = w;
    in.height = h;
    // printf("input===> %d %d \n",in.width, in.height);
    // int _size = in.width * in.height * 4;
    // printf("rgb ==> first channel => %d %d %d \n", in.data[0], in.data[1], in.data[2]);
    // rgb2bgr(in.data, _size);
    // printf("rgb ==> bgr first channel => %d %d %d \n", in.data[0], in.data[1], in.data[2]);
    img.w = ori_img_w;
    img.h = ori_img_h;
    img.c = 3;
    img.w_stride = ori_img_w * 3;
    img.data = imagedata;
    fill_dim(in.data, w, h);
}

void read_nv12(FrameInfo &in, const std::string &img_path, int img_w, int img_h) {
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
        return;
    }
    in.data = nv12_data;
    in.width = img_w;
    in.height = img_h;
}

void prepare_data(FrameInfo &in, magik::uranus::Tensor &out) {
    int in_w = in.width;
    int in_h = in.height;
    // printf("perpare_data, in_h in_w %d %d\n", in.height, in.width);
	magik::uranus::shape_t in_shape;
	magik::uranus::DataFormat data_format;
    if (DATA_FORMAT == magik::uranus::ChannelLayout::NV12) {
        in_shape = {1, in_h, in_w, 1};
		data_format = magik::uranus::DataFormat::NV12;
        
    } else {
        in_shape = {1, in_h, in_w, 4};
		data_format = magik::uranus::DataFormat::NHWC;
    }
	auto in_tensor = magik::uranus::Tensor(in_shape, magik::uranus::DataType::UINT8, data_format);
	uint8_t *in_tensor_data = in_tensor.mudata<uint8_t>();
	memcpy((void *)in_tensor_data, (void *)in.data, in_tensor.get_bytes_size());

    // resize
    magik::uranus::BsCommonParam param;
    param.pad_val = 114;
    param.pad_type = magik::uranus::BsPadType::SYMMETRY;
    param.in_layout = DATA_FORMAT;
    param.out_layout = magik::uranus::ChannelLayout::RGBA;
    param.input_height = in_h;
    param.input_width = in_w;
    param.input_line_stride = (DATA_FORMAT == magik::uranus::ChannelLayout::NV12) ? in_w : in_w * 4;
    param.addr_attr.vir_addr = in_tensor_data;
    vector<magik::uranus::Box_t> output_tmp;
    magik::uranus::Box_t tmp_box{0, 0, static_cast<float>(in_w), static_cast<float>(in_h)};
    output_tmp.push_back(tmp_box);
    // printf("work_h:%d work_w:%d \n", input_h, input_w);
    out.reshape({1, input_h, input_w, 4});
    uint8_t *out_data = out.mudata<uint8_t>();
    vector<magik::uranus::Tensor> output_tesnor;
    output_tesnor.push_back(out);
    magik::uranus::crop_common_resize(output_tesnor, output_tmp, magik::uranus::AddressLocate::NMEM_VIRTUAL, &param);

    in_tensor.free_data();

    if (IS_ALIGN_64(out_data) != 0) {
        fprintf(stderr, "input addr not align to 64 bytes.\n");
        return ;
    }
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

void run_model(magik::uranus::Network *model, magik::uranus::Tensor &img, std::vector<magik::uranus::Tensor> &outputs) {
    int img_size = img.get_bytes_size();
    auto input_names = model->get_input_names();
    for(auto n: input_names) {
        std::cout<<"input_names: "<< n <<std::endl;
        auto input = model->get_input(n);
        auto shape = input.shape();
        std::cout<<"input_shape: "<< shape[0] << " " << shape[1] << " " << shape[2] << " " << shape[3] << " " <<std::endl;

        uint8_t *indata = (uint8_t*)(input.vdata<uint8_t>());
        memcpy((void *)indata, (void *)img.vdata<uint8_t>(), img_size * sizeof(uint8_t));

        // const unsigned char* in_ptr = input.vdata<uint8_t>();
        // FILE *fp = fopen(std::string(n+".res").c_str(), "wb");
        // if (fp) {
        //     fwrite(in_ptr, 1, (input.get_bytes_size()), fp);
        //     fclose(fp);
        // } else {
        //     printf("open %s file failed\n", std::string(n+".res").c_str());
        //     return;
        // }
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
        auto shape = output.shape();
        std::cout<<"output_shape: "<< shape[0] << " " << shape[1] << " " << shape[2] << " " << shape[3] << " " <<std::endl;

        outputs.push_back(output);

        // const unsigned char* out_ptr = output.vdata<uint8_t>();
        // FILE *fp = fopen(std::string(n+".res").c_str(), "wb");
        // if (fp) {
        //     fwrite(out_ptr, 1, (output.get_bytes_size()), fp);
        //     fclose(fp);
        // } else {
        //     printf("open %s file failed\n", std::string(n+".res").c_str());
        //     return;
        // }
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
    magik::uranus::shape_t working_img_shape = {1, input_h, input_w, 4};
    magik::uranus::Tensor working_img(working_img_shape);
    printf("load image: %s \n", img_path.c_str());
    FrameInfo input_info;
    Img img;

    if(DATA_FORMAT == magik::uranus::ChannelLayout::NV12) {
        printf("read_nv12=========\n");
        read_nv12(input_info, img_path.c_str(), img_w, img_h);
    } else {
        printf("read_img=========\n");
        read_img(input_info, img, img_path.c_str());
    }
    RET_CHECKER(input_info.data != nullptr, "Read image failed!!!!");
    prepare_data(input_info, working_img);
    /******************model forward***********************/

    /** Gets the size of the memory space required for the network forward **/
    int ret = model->get_forward_memory_size();
    std::cout << "model forward memory size: " << ret << " = " << ret / 1024 << "KB" << " = " << ret / 1024 / 1024 << "MB" << std::endl;

    std::vector<magik::uranus::Tensor> outputs;
    run_model(model, working_img, outputs);

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
        Point pt1 = {
            .x = (int)output_box.box.x0,
            .y = (int)output_box.box.y0};
        Point pt2 = {
            .x = (int)output_box.box.x1,
            .y = (int)output_box.box.y1};
        if (DATA_FORMAT == magik::uranus::ChannelLayout::RGBA) {
            sample_draw_box_for_image(&img, pt1, pt2, color, 2);
        }
    }
    if (DATA_FORMAT == magik::uranus::ChannelLayout::RGBA) {
        stbi_write_bmp("result.bmp", img.w, img.h, 3, img.data); // w h
        free(img.data);
    }

    free(input_info.data);
    working_img.free_data();

    /******************destroy model***********************/
    magik_destroy_model(model);
    printf("magik_destroy_model over\n");
	std::cout << "^_^ Run successfully ^_^" << std::endl;
    return 0;
}

int main(int argc, char **argv) {
    std::string model_path = argv[1];
    std::string img_path = argv[2];

    /******************get_model_custom_info***********************/

    // int info_byte_size = 33;
    // char* info = (char*)malloc(info_byte_size);
    // memset(info, 0, info_byte_size);
    
    // if(get_model_custom_info(info, argv[1])==0){
    //     for(int i=0; i<info_byte_size; i++){
    //         printf("%c",info[i]);
    //     }
    //     printf("\n");
    // }else{
    //     printf("model has no info custom information!\n");
    // }

    /******************run detect***********************/
    if(DATA_FORMAT == magik::uranus::ChannelLayout::NV12) {
        if (argc != 5)
        {
            printf("%s model_path nv12_path w h\n", argv[0]);
            exit(0);
        }
        ori_img_w = atoi(argv[3]);
        ori_img_h = atoi(argv[4]);
        run_detect(model_path,img_path,ori_img_w,ori_img_h);
    } else{
        if (argc != 3)
        {
            printf("%s model_path rgba_path\n", argv[0]);
            exit(0);
        }
        ori_img_w = -1;
        ori_img_h = -1;
        run_detect(model_path,img_path,ori_img_w,ori_img_h);
    }
    return 0;
}
