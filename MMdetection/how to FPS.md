## Công thức quy đổi giữa FPS và thời gian xử lý ảnh

<img width="609" alt="image" src="https://github.com/user-attachments/assets/ebc879dd-6401-4cc1-a45d-11898153fbf3" />

### Cần tổ chức thư mục:

```
E:\thanh\ntu_group\thinh\  
|____ObjectDetection\  
|________mmdetection\  
|___________data\  
|_____________coco\  
|________________train2017\  
|________________val2017\    
|________________test2017\  
|________________annotations\  
|___________________instances_train2017.json  
|___________________instances_val2017.json  
|___________________instances_test2017.json  
|___________configs\  
|______________\efficientnet\  
|__________________retinanet_effb3_fpn_8xb4-crop896-1x_coco.py  
|___________checkpoints\  
|______________retina_ip102\  
|__________________epoch_98.pth
```

### Tính toán FPS

#### RetinaNet
Có thể thay đổi: --repeat-num 200 --num-warmup 50 thành 1, 1 cho nhanh

```
python tools\analysis_tools\benchmark.py configs\efficientnet\retinanet_effb3_fpn_8xb4-crop896-1x_coco.py --checkpoint checkpoints\retina_ip102\epoch_98.pth --task inference --repeat-num 1 --num-warmup 1
```
06/04 22:16:38 - mmengine - INFO - Overall fps: 14.3 img/s, times per image: 69.9 ms/img  
06/04 22:16:38 - mmengine - INFO - cuda memory: 76 MB  
06/04 22:16:39 - mmengine - INFO - (GB) mem_used: 17.72 | uss: 1.08 | total_proc: 1  

#### FasterR-CNN
```
python tools\analysis_tools\benchmark.py configs\faster_rcnn\faster-rcnn_r50_fpn_1x_coco.py --checkpoint checkpoints\frcnn_ip102\epoch_82.pth --task inference --repeat-num 1 --num-warmup 1
```
06/04 22:27:08 - mmengine - INFO - Overall fps: 42.6 img/s, times per image: 23.5 ms/img  
06/04 22:27:08 - mmengine - INFO - cuda memory: 171 MB  
06/04 22:27:08 - mmengine - INFO - (GB) mem_used: 17.75 | uss: 1.06 | total_proc: 1  
#### CascadeR-CNN
```
python tools\analysis_tools\benchmark.py configs\cascade_rcnn\cascade-rcnn_r50_fpn_1x_coco.py --checkpoint checkpoints\cascade_ip102\epoch_46.pth --task inference --repeat-num 1 --num-warmup 1
```
06/04 22:31:35 - mmengine - INFO - Overall fps: 34.2 img/s, times per image: 29.2 ms/img  
06/04 22:31:35 - mmengine - INFO - cuda memory: 278 MB  
06/04 22:31:36 - mmengine - INFO - (GB) mem_used: 17.71 | uss: 1.05 | total_proc: 1  

#### DynamicR-CNN
```
python tools\analysis_tools\benchmark.py configs\dynamic_rcnn/dynamic-rcnn_r50_fpn_1x_coco.py --checkpoint checkpoints\drcnn_ip102\epoch_22.pth --task inference --repeat-num 1 --num-warmup 1
```
06/04 22:44:33 - mmengine - INFO - Overall fps: 43.4 img/s, times per image: 23.0 ms/img
06/04 22:44:33 - mmengine - INFO - cuda memory: 171 MB
06/04 22:44:33 - mmengine - INFO - (GB) mem_used: 17.74 | uss: 1.06 | total_proc: 1
