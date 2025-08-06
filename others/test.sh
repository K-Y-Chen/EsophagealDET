python3 val.py --data oesopstomach_fusion.yaml --weight '/workspace/OesopStomach/code/v5.others/runs/train/centernetbase/weights/best.pt' --batch-size 32 --modelname centernet --name centernetbase

python3 val.py --data oesopstomach_fusion.yaml --weight '/workspace/OesopStomach/code/v5.others/runs/train/retinanetbase/weights/best.pt' --batch-size 32 --modelname retinanet --name retinanetbase &&

python3 val.py --data oesopstomach_fusion.yaml --weight '/workspace/OesopStomach/code/v5.others/runs/train/YOHObase/weights/best.pt' --batch-size 32 --modelname YOHO --name YOHObase &&

python3 val.py --data oesopstomach_fusion.yaml --weight '/workspace/OesopStomach/code/v5.others/runs/train/efficientdetbase/weights/best.pt' --batch-size 32 --modelname efficientdet --name efficientdetbase &&

python3 val.py --data oesopstomach_fusion.yaml --weight '/workspace/OesopStomach/code/v5.others/runs/train/detrbase/weights/best.pt' --batch-size 32 --modelname detr --name detrbase && 

echo done
