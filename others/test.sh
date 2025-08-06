python3 val.py --data oesopstomach_fusion.yaml --weight 'path_to_centernet_weights' --batch-size 32 --modelname centernet --name centernet

python3 val.py --data oesopstomach_fusion.yaml --weight 'path_to_retinanet_weights' --batch-size 32 --modelname retinanet --name retinanet &&

python3 val.py --data oesopstomach_fusion.yaml --weight 'path_to_YOHO_weights' --batch-size 32 --modelname YOHO --name YOHO &&

python3 val.py --data oesopstomach_fusion.yaml --weight ''path_to_efficientdet_weights' --batch-size 32 --modelname efficientdet --name efficientdet &&

python3 val.py --data oesopstomach_fusion.yaml --weight 'path_to_detr_weightst' --batch-size 32 --modelname detr --name detr && 

echo done


