# choose a root (example)
ROOT=/Users/justinbarry/projects/data/coco
mkdir -p "$ROOT" && cd "$ROOT"

# Images (train + val)
curl -L http://images.cocodataset.org/zips/train2017.zip -o train2017.zip
curl -L http://images.cocodataset.org/zips/val2017.zip   -o val2017.zip
unzip -q train2017.zip && rm train2017.zip
unzip -q val2017.zip   && rm val2017.zip

# Annotations (captions)
curl -L http://images.cocodataset.org/annotations/annotations_trainval2017.zip -o annotations_trainval2017.zip
unzip -q annotations_trainval2017.zip annotations/captions_train2017.json annotations/captions_val2017.json
rm annotations_trainval2017.zip