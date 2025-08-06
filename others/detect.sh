python3 detect.py --data ./data/oesopstomach_fusion.yaml --weight '/workspace/OesopStomach/code/v5.others/runs/train/centernetbase/weights/best.pt'  --modelname centernet --name centernetbase && 

python3 detect.py --data ./data/oesopstomach_fusion.yaml --weight '/workspace/OesopStomach/code/v5.others/runs/train/retinanetbase/weights/best.pt'  --modelname retinanet --name retinanetbase &&

python3 detect.py --data ./data/oesopstomach_fusion.yaml --weight '/workspace/OesopStomach/code/v5.others/runs/train/YOHObase/weights/best.pt'  --modelname YOHO --name YOHObase &&

python3 detect.py --data ./data/oesopstomach_fusion.yaml --weight '/workspace/OesopStomach/code/v5.others/runs/train/efficientdetbase/weights/best.pt'  --modelname efficientdet --name efficientdetbase &&

python3 detect.py --data ./data/oesopstomach_fusion.yaml --weight '/workspace/OesopStomach/code/v5.others/runs/train/detrbase/weights/best.pt'  --modelname detr --name detrbase && 

echo done