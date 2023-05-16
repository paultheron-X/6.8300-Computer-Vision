
# create a list of 000 to 029
list=$(seq -f "%03g" 0 29)

for i in $list
do
    # copy the folder at data/processed/val/val_sharp/$i to data/processed/train/train_sharp/ and rename is val_$i
    cp -r data/processed/val/val_sharp/$i data/processed/train/train_sharp/val_$i
    cp -r data/processed/val/val_sharp_bicubic/X4/$i data/processed/train/train_sharp_bicubic/X4/val_$i
done