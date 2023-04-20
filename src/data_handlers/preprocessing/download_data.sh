

echo "Downloading REDS data from website https://seungjunnah.github.io/Datasets/reds.html"

mkdir ./data/raw/train_orig

# Download REDS dataset
# dataset 0

wget -c http://data.cv.snu.ac.kr:8008/webdav/dataset/REDS/orig/train_orig_part0.zip -P ./data/raw

unzip ./data/raw/train_orig_part0.zip -d ./data/raw

cp ./data/raw/train_orig_part0/train_orig/* ./data/raw/train_orig

# dataset 1

wget -c http://data.cv.snu.ac.kr:8008/webdav/dataset/REDS/orig/train_orig_part1.zip -P ./data/raw

unzip ./data/raw/train_orig_part1.zip -d ./data/raw

cp ./data/raw/train_orig_part1/train_orig/* ./data/raw/train_orig


