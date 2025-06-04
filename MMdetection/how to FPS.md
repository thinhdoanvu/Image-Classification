## Công thức quy đổi giữa FPS và thời gian xử lý ảnh

**Từ thời gian xử lý (ms/ảnh) sang FPS**:

\[
\text{FPS} = \frac{1000}{\text{Time per image (ms)}}
\]

**Từ FPS sang thời gian xử lý (ms/ảnh)**:

\[
\text{Time per image (ms)} = \frac{1000}{\text{FPS}}
\]


python tools\analysis_tools\benchmark.py configs\efficientnet\retinanet_effb3_fpn_8xb4-crop896-1x_coco.py --checkpoint checkpoints\retina_ip102\epoch_98.pth --task inference --repeat-num 200 --num-warmup 50


  
