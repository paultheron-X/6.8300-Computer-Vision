# process data:

ln -s /Data/reds_dataset/raw/ data/raw

ln -s /Data/reds_dataset/processed/ data/processed

unzip /Data/reds_dataset/raw/val_sharp.zip -d /Data/reds_dataset/processed/val_sharp

unzip /Data/reds_dataset/raw/val_sharp_bicubic.zip -d /Data/reds_dataset/processed/val_sharp_bicubic

unzip /Data/reds_dataset/raw/train_sharp_bicubic.zip -d /Data/reds_dataset/processed/train_sharp_bicubic

unzip /Data/reds_dataset/raw/train_sharp.zip -d /Data/reds_dataset/processed/train_sharp

mkdir data/processed/train

mkdir data/processed/val

mv /Data/reds_dataset/processed/val_sharp/val/* data/processed/val

mv /Data/reds_dataset/processed/val_sharp_bicubic/val/* data/processed/val

mv /Data/reds_dataset/processed/train_sharp_bicubic/train/* data/processed/train

mv /Data/reds_dataset/processed/train_sharp/train/train_sharp data/processed/train
