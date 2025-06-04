## Công thức quy đổi giữa FPS và thời gian xử lý ảnh

<img width="609" alt="image" src="https://github.com/user-attachments/assets/ebc879dd-6401-4cc1-a45d-11898153fbf3" />

Cần tổ chức thư mục:
E:\thanh\ntu_group\thinh\
|____ObjectDetection\
|________mmdetection\
|___________data\
|_____________coco\
|________________train2017
|________________val2017  
|________________test2017
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
python tools\analysis_tools\benchmark.py configs\efficientnet\retinanet_effb3_fpn_8xb4-crop896-1x_coco.py --checkpoint checkpoints\retina_ip102\epoch_98.pth --task inference --repeat-num 200 --num-warmup 50
```

  
