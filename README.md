# HW5_Aidea-AOI-image-recognition
##How to Train ResNet50 for AOI image recognition

Step.0 使用 data_preprocess.py 將原始資料之標記資料集，分類至各自類別之資料夾。
Step.1 使用 aug_image1004.py 將原始資料之標記資料集，依照增生效果設定，增生十次。
Step.2 透過 resnet50_implement_onAOI.py，利用 Step.1 的增生資料與原始資料之標記資料集訓練一個ResNet50模型
