```
wget -O ./musiccaps-dataset.zip https://www.kaggle.com/api/v1/datasets/download/pupdown/musiccaps-dataset
unzip musiccaps-dataset.zip -d ../musiccaps
cat ../musiccaps/music_data.tar.bz2.* > ../musiccaps/music_data.tar.bz2
tar -xjvf ../musiccaps/music_data.tar.bz2 -C ../musiccaps

wget -O ./musiccaps-csv.zip https://www.kaggle.com/api/v1/datasets/download/googleai/musiccaps
unzip musiccaps-csv.zip -d ../musiccaps
```