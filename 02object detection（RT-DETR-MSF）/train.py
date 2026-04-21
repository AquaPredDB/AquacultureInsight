import warnings, os

warnings.filterwarnings('ignore')
from ultralytics import RTDETR


if __name__ == '__main__':
    model = RTDETR('ultralytics/cfg/models/rt-detr/RT-DETR-MSF.yaml')
    # model.load('') # loading pretrain weights
    model.train(data='dataset/data.yaml',
                cache=False,
                imgsz=640,
                epochs=200,
                batch=4, 
                workers=4, 
                device='0,1', 
                resume='',
                project='runs/train',
                name='exp',
                )
