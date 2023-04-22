
echo "Downloading REDS data from website https://seungjunnah.github.io/Datasets/reds.html"

mkdir -p /Data/reds_dataset/raw/train_orig


for i in {0..15}; do
    echo "Prep part $i"
    scp /Data/reds_dataset/raw/train_orig_part$i.zip paul.theron@bentley.polytechnique.fr:/Data/reds_dataset/raw/
    #wget -c http://data.cv.snu.ac.kr:8008/webdav/dataset/REDS/orig/train_orig_part$i.zip -P /Data/reds_dataset/raw

    #unzip /Data/reds_dataset/raw/train_orig_part$i.zip -d /Data/reds_dataset/raw

    #mv /Data/reds_dataset/raw/train/train_orig/* /Data/reds_dataset/raw/train_orig

    #rm -rf /Data/reds_dataset/raw/train_orig_part$i
done