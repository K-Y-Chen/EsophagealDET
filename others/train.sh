python3 train.py --data oesopstomach_fusion.yaml --epochs 500 --weight '' --cfg yolov5n.yaml --batch-size 32 --model centernet --name centernetbase && 

python3 train.py --data oesopstomach_fusion.yaml --epochs 500 --weight '' --cfg yolov5n.yaml --batch-size 32 --model retinanet --name retinanetbase &&

python3 train.py --data oesopstomach_fusion.yaml --epochs 500 --weight '' --cfg yolov5n.yaml --batch-size 32 --model YOHO --name YOHObase &&

python3 train.py --data oesopstomach_fusion.yaml --epochs 500 --weight '' --cfg yolov5n.yaml --batch-size 32 --model efficientdet --name efficientdetbase &&

python3 train.py --data oesopstomach_fusion.yaml --epochs 500 --weight '' --cfg yolov5n.yaml --batch-size 2 --model detr --name detrbase && 


echo done

