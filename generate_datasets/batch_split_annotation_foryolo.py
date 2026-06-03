import os
import subprocess
import sys
import shutil
import argparse

parse = argparse.ArgumentParser()
parse.add_argument("--cocodir", help="coco dataset path")
parse.add_argument("--outdir", help="output-dir")
parse.add_argument("--cocoid", help="ID of coco data you need to extract: 1,2,3")
args = parse.parse_args()


CURDIR = os.path.dirname(os.path.realpath(__file__))
print(CURDIR)
redo = True
coco_id = args.cocoid

coco_data_dir = args.cocodir
anno_sets = ["instances_train2017", "instances_val2017", "image_info_test2017"]
anno_dir = "{}/annotations".format(coco_data_dir)
out_anno_dir = args.outdir
imgset_dir = "{}/ImageSets".format(out_anno_dir)

####creat labels
for i in range(0, len(anno_sets)):
    anno_set = anno_sets[i]
    anno_file = "{}/{}.json".format(anno_dir, anno_set)
    if not os.path.exists(anno_file):
        print("{} does not exist".format(anno_file))
        continue
    anno_name = anno_set.split("_")[-1]
    out_dir = "{}/labels_{}".format(out_anno_dir, anno_set)
    imgset_file = "{}/{}.txt".format(imgset_dir, anno_name)
    print(out_dir, imgset_file, anno_file)
    if redo or not os.path.exists(out_dir):
        cmd = "python {}/split_annotation_foryolo.py --out-dir={} --imgset-file={} {} --cocoid={}" \
                .format(CURDIR, out_dir, imgset_file, anno_file, coco_id)
        process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
        output = process.communicate()[0]

