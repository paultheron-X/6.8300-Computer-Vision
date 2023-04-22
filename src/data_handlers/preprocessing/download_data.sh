

echo "Downloading REDS data from website https://seungjunnah.github.io/Datasets/reds.html"

mkdir -p /Data/reds_dataset/raw/train_orig


for i in {9..15}; do
    echo "Downloading part $i"
    wget -c http://data.cv.snu.ac.kr:8008/webdav/dataset/REDS/orig/train_orig_part$i.zip -P /Data/reds_dataset/raw

    #unzip /Data/reds_dataset/raw/train_orig_part$i.zip -d /Data/reds_dataset/raw

    #cp /Data/reds_dataset/raw/train_orig_part$i/train_orig/* /Data/reds_dataset/raw/train_orig
done