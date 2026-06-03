from pycocotools.coco import COCO
import argparse
from collections import OrderedDict
import json
import os
from pprint import pprint
import sys
import matplotlib.pyplot as plt
import pylab
sys.path.append(os.path.dirname(sys.path[0]))
pylab.rcParams['figure.figsize'] = (8.0, 10.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Split whole json annotation to individual files.")
    parser.add_argument("annofile",
            help = "The file which contains all the annotations for a dataset in json format.")
    parser.add_argument("--out-dir", default = "",
            help = "The output directory where we store the annotation per image.")
    parser.add_argument("--imgset-file", default = "",
            help = "A file where we store all the image names of a dataset.")
    parser.add_argument("--cocoid", type = str,
            help = "ID of coco data you need to extract")

    def is_empty_file(file_path: str):
        assert isinstance(file_path, str)
        assert os.path.isfile(file_path)
        return os.path.getsize(file_path) == 0

    args = parser.parse_args()
    annofile = args.annofile
    if not os.path.exists(annofile):
        print("{} does not exist!".format(annofile))
        sys.exit()

    out_dir = args.out_dir
    train_data = "train" in out_dir
    if out_dir:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
    imgset_file = args.imgset_file
    if imgset_file:
        imgset_dir = os.path.dirname(imgset_file)
        if not os.path.exists(imgset_dir):
            os.makedirs(imgset_dir)

    # initialize COCO api.
    coco = COCO(annofile)

    img_ids = coco.getImgIds()
    img_names = []
    for img_id in img_ids:
        # get image info
        img = coco.loadImgs(img_id)
        file_name = img[0]["file_name"]
        name = os.path.splitext(file_name)[0]
        if out_dir:
            # get annotation info
            anno_ids = coco.getAnnIds(imgIds=img_id, iscrowd=None)
            #anno_ids = coco.getAnnIds(imgIds=img_id, iscrowd=1)
            anno = coco.loadAnns(anno_ids)
            # save annotation to file
            img_anno = dict()
            img_anno["image"] = img[0]
            img_anno["annotation"] = anno
            width = img[0]["width"]
            height = img[0]["height"]
            dw = 1. / (width)
            dh = 1. / (height)
            yoloanno_filename = "{}/{}.txt".format(out_dir, name)

            write_txt = False
            if not train_data:
                yoloanno_file = open(yoloanno_filename, 'w')
                write_txt = True
            names = {}
            cocoid = args.cocoid.split(",")
            for index,value in enumerate(cocoid):
                names[int(value)] =index

            is_fist = True

            if(len(anno) > 0):
                for s_anno in anno:
                    x_center = s_anno["bbox"][0]+s_anno["bbox"][2]/2.0
                    y_center = s_anno["bbox"][1]+s_anno["bbox"][3]/2.0
                    w = s_anno["bbox"][2]
                    h = s_anno["bbox"][3]
                    x_center = x_center * dw
                    y_center = y_center * dh
                    w = w * dw
                    h = h * dh
                    if s_anno["category_id"] in names.keys():
                        if is_fist and train_data:
                            yoloanno_file = open(yoloanno_filename, 'w')
                            is_fist = False
                            write_txt = True
                        cat_id = names[s_anno["category_id"]]
                        bb = (x_center, y_center, w, h)
                        yoloanno_file.write(str(cat_id) + " " + " ".join([str(a) for a in bb]) + '\n')
            if imgset_file and write_txt:
                name = "/data/coco2017/images/"+name+".jpg"
                img_names.append(name)
    if img_names:
        img_names.sort()
        with open(imgset_file, "w") as f:
            f.write("\n".join(img_names))
