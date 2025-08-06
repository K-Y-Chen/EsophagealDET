python3 detect.py --data ./data/oesopstomach_fusion.yaml --weight 'path_to_centernet_weights'  --modelname centernet --name centernet && 

python3 detect.py --data ./data/oesopstomach_fusion.yaml --weight 'path_to_retinanet_weights'  --modelname retinanet --name retinanet &&

python3 detect.py --data ./data/oesopstomach_fusion.yaml --weight 'path_to_YOHO_weights'  --modelname YOHO --name YOHObase &&

python3 detect.py --data ./data/oesopstomach_fusion.yaml --weight 'path_to_efficientdet_weights'  --modelname efficientdet --name efficientdet &&

python3 detect.py --data ./data/oesopstomach_fusion.yaml --weight 'path_to_detr_weightst'  --modelname detr --name detr && 

echo done
