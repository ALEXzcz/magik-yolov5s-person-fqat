export NCCL_IB_DISABLE=1
export NCCL_DEBUG=info

###float32:
# python3 train.py \
#     --data data/coco-person.yaml\
#     --cfg models/yolov5s.yaml \
#     --weights '' \
#     --batch-size 16 \
#     --hyp data/hyp.scratch-float.yaml \
#     --project ./runs/train/yolov5s-person-float32 \
#     --epochs 30 \
#     --device 0 \
#     --bit 32

###4bit coco-person:
# python3 train.py \
#     --data data/coco-person.yaml\
#     --cfg models/yolov5s.yaml \
#     --weights 'checkpoint/32W32F/weights/best.pt' \
#     --batch-size 16 \
#     --hyp data/hyp.scratch.yaml \
#     --project ./runs/train/yolov5s-person-4bit \
#     --epochs 300 \
#     --device 0 \
#     --optimizer Adam \
#     --bit 4

# 4bit anytrek_2310:
python3 train.py \
    --data data/anytrek_2310.yaml\
    --cfg models/yolov5s.yaml \
    --weights 'checkpoint/32W32F/weights/best.pt' \
    --batch-size 16 \
    --hyp data/hyp.scratch.yaml \
    --project "$OUT_DIR" \
    --epochs 100 \
    --device 0 \
    --optimizer Adam \
    --bit 4

mkdir -p "$OUT_WEIGHTS_DIR"
cp "$OUT_DIR/weights/best.pt" "$OUT_WEIGHTS_DIR/weights/best-tmp.pt"

#checkpoint/32W32F/weights/best.pt 32bit lr 0.01 sgd
#checkpoint/8W8F/weights/best.pt 32bit magi_quantize.py --> 8bit
#checkpoint/4W4F/weights/best.pt 4bit lr 0.001 adam
