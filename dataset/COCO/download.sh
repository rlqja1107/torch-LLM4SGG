
if [ $# -eq 0 ]; then
  echo "Please provide the destination folder as an argument."
  exit 1
fi

dest_folder="$1"
echo $dest_folder
cd "$dest_folder"
wget -N http://images.cocodataset.org/zips/val2017.zip

if test -e "val2017.zip"; then
    unzip val2017.zip
fi

if test -d "val2017"; then
    mv val2017 images
    rm -rf val2017
fi

wget -N http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip
mv train2017/* images

if test -e "train2017.zip"; then
    unzip train2017.zip
fi

if test -d "train2017"; then
    mv train2017/* images
    rm -rf train2017
fi

wget -N http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip
rm -rf annotations/person_keypoints_val2017.json
rm -rf annotations/person_keypoints_train2017.json
rm -rf annotations/instances_val2017.json
rm -rf annotations/instances_train2017.json

mv annotations/captions_train2017.json ./
mv annotations/captions_val2017.json ./


if test -e "train2017.zip"; then
    rm -rf annotations_trainval2017.zip
fi