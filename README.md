# Image-Classification
Data folder structure

![image](https://github.com/thinhdoanvu/Image-Classification/assets/22977443/b8e40b28-a20a-4211-bc85-9ff821965842)

Train data:

![image](https://github.com/thinhdoanvu/Image-Classification/assets/22977443/848e6099-982f-406b-8de8-7d943799a1cb)

Test data:

![image](https://github.com/thinhdoanvu/Image-Classification/assets/22977443/fbb719e7-70dc-493d-8c5e-c282a85a6a50)

Valid data:

![image](https://github.com/thinhdoanvu/Image-Classification/assets/22977443/fa277613-29b8-461b-9b84-412d3cb19db9)

### Download dataset
https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification/download?datasetVersionNumber=2

### CSV file:
class index	filepaths	labels	card type	data set

0	train/ace of clubs/001.jpg	ace of clubs	ace	train

1	train/ace of diamonds/001.jpg	ace of diamonds	ace	train

2	train/ace of hearts/001.jpg	ace of hearts	ace	train

3	train/ace of spades/001.jpg	ace of spades	ace	train
...

![image](https://github.com/thinhdoanvu/Image-Classification/assets/22977443/d35b7534-edf8-4a91-a207-a52c4ed9a6de)


### Testing with test folder
<img width="1104" alt="image" src="https://github.com/thinhdoanvu/Image-Classification/assets/22977443/18266de1-f53c-4f56-a3db-623b1b4acf1b">

The classify model has been loaded.

Image: 01.jpg, Predicted label: ace of clubs

Image: 02.jpg, Predicted label: eight of clubs

Image: 03.jpg, Predicted label: five of hearts

Image: 04.jpg, Predicted label: jack of hearts

Image: 05.jpg, Predicted label: jack of hearts

Image: 1.jpg, Predicted label: joker

Image: 3.jpg, Predicted label: king of clubs

Image: 4.jpg, Predicted label: four of clubs

#Train by alexnet model
Training: 100%|███████████████████████████████| 239/239 [00:46<00:00
Valid: 100%|██████████████████████████████████| 239/239 [00:23<00:00
Epoch:0 training loss is 3.6487
Epoch:0 valid loss is 15.5723
Training: 100%|███████████████████████████████| 239/239 [03:33<00:00
Valid: 100%|██████████████████████████████████| 239/239 [01:09<00:00
Epoch:1 training loss is 6.0976
Epoch:1 valid loss is 2.3439
Training: 100%|███████████████████████████████| 239/239 [03:34<00:00
Valid: 100%|██████████████████████████████████| 239/239 [01:09<00:00
Epoch:2 training loss is 1.9789
Epoch:2 valid loss is 1.9730
Training: 100%|███████████████████████████████| 239/239 [03:35<00:00
Valid: 100%|██████████████████████████████████| 239/239 [00:55<00:00
Epoch:3 training loss is 2.7794
Epoch:3 valid loss is 3.1462
Training: 100%|███████████████████████████████| 239/239 [03:27<00:00
Valid: 100%|██████████████████████████████████| 239/239 [01:08<00:00
Epoch:4 training loss is 2.4875
Epoch:4 valid loss is 3.1180
Training: 100%|███████████████████████████████| 239/239 [02:59<00:00
Valid: 100%|██████████████████████████████████| 239/239 [00:18<00:00
Epoch:5 training loss is 0.7570
Epoch:5 valid loss is 0.5528
Training: 100%|███████████████████████████████| 239/239 [00:46<00:00
Valid: 100%|██████████████████████████████████| 239/239 [00:18<00:00
Epoch:6 training loss is 1.5866
Epoch:6 valid loss is 1.2579
Training: 100%|███████████████████████████████| 239/239 [00:46<00:00
Valid: 100%|██████████████████████████████████| 239/239 [00:18<00:00
Epoch:7 training loss is 0.4911
Epoch:7 valid loss is 0.2185
Training: 100%|███████████████████████████████| 239/239 [00:47<00:00
Valid: 100%|██████████████████████████████████| 239/239 [00:18<00:00
Epoch:8 training loss is 0.2505
Epoch:8 valid loss is 2.1068
Training: 100%|███████████████████████████████| 239/239 [00:46<00:00
Valid: 100%|██████████████████████████████████| 239/239 [00:18<00:00
Epoch:9 training loss is 1.7515
Epoch:9 valid loss is 0.5408
Training: 100%|███████████████████████████████| 239/239 [00:46<00:00
Valid: 100%|██████████████████████████████████| 239/239 [00:18<00:00
Epoch:10 training loss is 1.7918
Epoch:10 valid loss is 0.5783
Training: 100%|███████████████████████████████| 239/239 [00:47<00:00
Valid: 100%|██████████████████████████████████| 239/239 [00:18<00:00
Epoch:11 training loss is 7.8102
Epoch:11 valid loss is 0.4725
Training: 100%|███████████████████████████████| 239/239 [00:47<00:00
Valid: 100%|██████████████████████████████████| 239/239 [00:18<00:00
Epoch:12 training loss is 0.4697
Epoch:12 valid loss is 0.0213
Training: 100%|███████████████████████████████| 239/239 [00:47<00:00
Valid: 100%|██████████████████████████████████| 239/239 [00:18<00:00
Epoch:13 training loss is 0.5176
Epoch:13 valid loss is 0.0236
Training: 100%|███████████████████████████████| 239/239 [00:47<00:00
Valid: 100%|██████████████████████████████████| 239/239 [00:18<00:00
Epoch:14 training loss is 1.5612
Epoch:14 valid loss is 5481.7866
Training: 100%|███████████████████████████████| 239/239 [00:47<00:00
Valid: 100%|██████████████████████████████████| 239/239 [00:18<00:00
Epoch:15 training loss is 0.1492
Epoch:15 valid loss is 0.2685
Training: 100%|███████████████████████████████| 239/239 [00:47<00:00
Valid: 100%|██████████████████████████████████| 239/239 [00:18<00:00
Epoch:16 training loss is 0.4390
Epoch:16 valid loss is 0.0817
Training: 100%|███████████████████████████████| 239/239 [00:47<00:00
Valid: 100%|██████████████████████████████████| 239/239 [00:18<00:00
Epoch:17 training loss is 1.2166
Epoch:17 valid loss is 0.1830
Training: 100%|███████████████████████████████| 239/239 [00:47<00:00
Valid: 100%|██████████████████████████████████| 239/239 [00:18<00:00
Epoch:18 training loss is 1.4200
Epoch:18 valid loss is 0.4292
Training: 100%|███████████████████████████████| 239/239 [00:47<00:00
Valid: 100%|██████████████████████████████████| 239/239 [00:18<00:00
Epoch:19 training loss is 1.3585
Epoch:19 valid loss is 0.0006
Training: 100%|███████████████████████████████| 239/239 [00:47<00:00
Valid: 100%|██████████████████████████████████| 239/239 [00:18<00:00
Epoch:20 training loss is 2.2221
Epoch:20 valid loss is 0.0406
Training: 100%|███████████████████████████████| 239/239 [00:47<00:00
Valid: 100%|██████████████████████████████████| 239/239 [00:18<00:00
Epoch:21 training loss is 0.1233
Epoch:21 valid loss is 0.1290
Training: 100%|███████████████████████████████| 239/239 [00:47<00:00
Valid: 100%|██████████████████████████████████| 239/239 [00:18<00:00
Epoch:22 training loss is 2.4088
Epoch:22 valid loss is 0.0001
Training: 100%|███████████████████████████████| 239/239 [00:47<00:00
Valid: 100%|██████████████████████████████████| 239/239 [00:18<00:00
Epoch:23 training loss is 0.0968
Epoch:23 valid loss is 0.0000
Training: 100%|███████████████████████████████| 239/239 [00:47<00:00
Valid: 100%|██████████████████████████████████| 239/239 [00:18<00:00
Epoch:24 training loss is 0.0019
Epoch:24 valid loss is 0.0009
Training: 100%|███████████████████████████████| 239/239 [00:47<00:00
Valid: 100%|██████████████████████████████████| 239/239 [00:18<00:00
Epoch:25 training loss is 0.1196
Epoch:25 valid loss is 0.0009
Training: 100%|███████████████████████████████| 239/239 [00:47<00:00
Valid: 100%|██████████████████████████████████| 239/239 [00:18<00:00
Epoch:26 training loss is 0.0593
Epoch:26 valid loss is 0.4260
Training: 100%|███████████████████████████████| 239/239 [00:47<00:00
Valid: 100%|██████████████████████████████████| 239/239 [00:18<00:00
Epoch:27 training loss is 0.0076
Epoch:27 valid loss is 0.0000
Training: 100%|███████████████████████████████| 239/239 [00:47<00:00
Valid: 100%|██████████████████████████████████| 239/239 [00:18<00:00
Epoch:28 training loss is 0.4046
Epoch:28 valid loss is 0.0163
Training: 100%|███████████████████████████████| 239/239 [00:47<00:00
Valid: 100%|██████████████████████████████████| 239/239 [00:18<00:00
Epoch:29 training loss is 0.0052
Epoch:29 valid loss is 0.0008
Training: 100%|███████████████████████████████| 239/239 [00:47<00:00
Valid: 100%|██████████████████████████████████| 239/239 [00:18<00:00
Epoch:30 training loss is 0.6516
Epoch:30 valid loss is 0.0044
Training: 100%|███████████████████████████████| 239/239 [00:47<00:00
Valid: 100%|██████████████████████████████████| 239/239 [00:18<00:00
Epoch:31 training loss is 0.0560
Epoch:31 valid loss is 0.0013
Training: 100%|███████████████████████████████| 239/239 [00:47<00:00
Valid: 100%|██████████████████████████████████| 239/239 [00:18<00:00
Epoch:32 training loss is 0.0009
Epoch:32 valid loss is 0.2832
Training: 100%|███████████████████████████████| 239/239 [00:47<00:00
Valid: 100%|██████████████████████████████████| 239/239 [00:18<00:00
Epoch:33 training loss is 0.2726
Epoch:33 valid loss is 94.9121
Training: 100%|███████████████████████████████| 239/239 [00:47<00:00
Valid: 100%|██████████████████████████████████| 239/239 [00:18<00:00
Epoch:34 training loss is 0.0038
Epoch:34 valid loss is 0.0010
Training: 100%|███████████████████████████████| 239/239 [00:47<00:00
Valid: 100%|██████████████████████████████████| 239/239 [00:18<00:00
Epoch:35 training loss is 0.0280
Epoch:35 valid loss is 0.0054
Training: 100%|███████████████████████████████| 239/239 [00:47<00:00
Valid: 100%|██████████████████████████████████| 239/239 [00:18<00:00
Epoch:36 training loss is 0.0179
Epoch:36 valid loss is 0.0001
Training: 100%|███████████████████████████████| 239/239 [00:47<00:00
Valid: 100%|██████████████████████████████████| 239/239 [00:18<00:00
Epoch:37 training loss is 0.0008
Epoch:37 valid loss is 0.0850
Training: 100%|███████████████████████████████| 239/239 [00:47<00:00
Valid: 100%|██████████████████████████████████| 239/239 [00:18<00:00
Epoch:38 training loss is 0.0002
Epoch:38 valid loss is 0.1479
Training: 100%|███████████████████████████████| 239/239 [00:47<00:00
Valid: 100%|██████████████████████████████████| 239/239 [00:18<00:00
Epoch:39 training loss is 0.0694
Epoch:39 valid loss is 0.0000
Training: 100%|███████████████████████████████| 239/239 [00:47<00:00
Valid: 100%|██████████████████████████████████| 239/239 [00:18<00:00
Epoch:40 training loss is 0.0933
Epoch:40 valid loss is 0.0000
Training: 100%|███████████████████████████████| 239/239 [00:47<00:00
Valid: 100%|██████████████████████████████████| 239/239 [00:18<00:00
Epoch:41 training loss is 0.0033
Epoch:41 valid loss is 0.0003
Training: 100%|███████████████████████████████| 239/239 [00:47<00:00
Valid: 100%|██████████████████████████████████| 239/239 [00:18<00:00
Epoch:42 training loss is 0.7724
Epoch:42 valid loss is 0.0078
Training: 100%|███████████████████████████████| 239/239 [00:47<00:00
Valid: 100%|██████████████████████████████████| 239/239 [00:18<00:00
Epoch:43 training loss is 1.5035
Epoch:43 valid loss is 0.0060
Training: 100%|███████████████████████████████| 239/239 [00:47<00:00
Valid: 100%|██████████████████████████████████| 239/239 [00:18<00:00
Epoch:44 training loss is 0.8187
Epoch:44 valid loss is 0.0000
Training: 100%|███████████████████████████████| 239/239 [00:47<00:00
Valid: 100%|██████████████████████████████████| 239/239 [00:18<00:00
Epoch:45 training loss is 0.0011
Epoch:45 valid loss is 0.0001
Training: 100%|███████████████████████████████| 239/239 [00:47<00:00
Valid: 100%|██████████████████████████████████| 239/239 [00:18<00:00
Epoch:46 training loss is 0.0080
Epoch:46 valid loss is 0.0023
Training: 100%|███████████████████████████████| 239/239 [00:47<00:00
Valid: 100%|██████████████████████████████████| 239/239 [00:18<00:00
Epoch:47 training loss is 0.0032
Epoch:47 valid loss is 0.0006
Training: 100%|███████████████████████████████| 239/239 [00:47<00:00
Valid: 100%|██████████████████████████████████| 239/239 [00:18<00:00
Epoch:48 training loss is 0.2076
Epoch:48 valid loss is 0.0000
Training: 100%|███████████████████████████████| 239/239 [00:47<00:00
Valid: 100%|██████████████████████████████████| 239/239 [00:18<00:00
Epoch:49 training loss is 0.0021
Epoch:49 valid loss is 0.0000

# Pretrain weights
## imagenet18:
![pretrained_weights_resnet18_imagenet](https://github.com/thinhdoanvu/Image-Classification/assets/22977443/2d452b4f-c49c-4b8c-b806-279f52a8b362)

## without pretrain weight: 
outputs folder
![pretrained_weights_epoch_99](https://github.com/thinhdoanvu/Image-Classification/assets/22977443/269f14ed-0e1b-449a-9e66-762d93e2dfb2)
