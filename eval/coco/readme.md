To run the config, replace the line
```
WEIGHTS: "" # replace weight folder
```
with correct weight path in `config_coco.yaml`

Then run `python train.py config_coco.yaml` to launch coco training.

The weights need to be converted using the official [conversion script](https://github.com/facebookresearch/detectron2/blob/main/tools/convert-torchvision-to-d2.py)