export NCCL_IB_DISABLE=1
export NCCL_DEBUG=info

###float32:
# python3 train.py \
#     --data data/coco-person.yaml\
#     --cfg models/yolov5s.yaml \
#     --weights '' \
#     --batch-size 64 \
#     --hyp data/hyp.scratch-float.yaml \
#     --project ./runs/train/yolov5s-person-float32 \
#     --epochs 30 \
#     --device 0 \
#     --bit 32

###4bit:
python3 train.py \
    --data data/coco-person.yaml\
    --cfg models/yolov5s.yaml \
    --weights 'checkpoint/32W32F/weights/best.pt' \
    --batch-size 64 \
    --hyp data/hyp.scratch.yaml \
    --project ./runs/train/yolov5s-person-4bit \
    --epochs 300 \
    --device 0 \
    --optimizer Adam \
    --bit 4

#checkpoint/32W32F/weights/best.pt 32bit lr 0.01 sgd
#checkpoint/8W8F/weights/best.pt 32bit magi_quantize.py --> 8bit
#checkpoint/4W4F/weights/best.pt 4bit lr 0.001 adam
