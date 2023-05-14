
# validation data:

#wget -c http://data.cv.snu.ac.kr:8008/webdav/dataset/REDS/val_sharp.zip -P /Data/reds_dataset/raw

#wget -c http://data.cv.snu.ac.kr:8008/webdav/dataset/REDS/val_sharp_bicubic.zip -P /Data/reds_dataset/raw

#wget -c http://data.cv.snu.ac.kr:8008/webdav/dataset/REDS/train_sharp_bicubic.zip -P /Data/reds_dataset/raw

#wget -c http://data.cv.snu.ac.kr:8008/webdav/dataset/REDS/train_sharp.zip -P /Data/reds_dataset/raw

scp /Data/reds_dataset/raw/val_sharp.zip paul.theron@mazda.polytechnique.fr:/Data/reds_dataset/raw/
scp /Data/reds_dataset/raw/val_sharp_bicubic.zip paul.theron@mazda.polytechnique.fr:/Data/reds_dataset/raw/
scp /Data/reds_dataset/raw/train_sharp_bicubic.zip paul.theron@mazda.polytechnique.fr:/Data/reds_dataset/raw/
scp /Data/reds_dataset/raw/train_sharp.zip paul.theron@mazda.polytechnique.fr:/Data/reds_dataset/raw/