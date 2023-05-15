# ViT

Vision transformer aka ViT is just an extension of transformers to computer vision. Standard Transformer consists of encoder and decoder blocks. But here we use encoder part only.When compared to CNNs, ViT requires fewer computational resources to train. ViT first appeared in a paper called "An Image is worth 16x16 words: Transformers for image recognition at scale". This paper proved that reliance on CNN is not necessary. We can directly give images to standard transformer and it'll perform classification task better than CNNs did. But some preprocessing is required as standard transformer accepts input as word tokens. Similar to word token here we split image into patches.

## About this implementation

For this project, I used a [git repo](https://github.com/faustomorales/vit-keras) which is implemented using TensorFlow as a reference. This reference repo and the original implementation, both used multi-head attention mechanism. But I chose single-head attention instead. Weights of the model are saved from reference repo, and the [forked version](https://github.com/kavysabu1996/vit-keras) contains code for the same.

## image classification using this repo

create a python virtual environment and install all requirments

virtual env creation - `Ubuntu`
```
# in root directory
sudo apt update
sudo apt install python3-dev python3-pip python3-venv
python3 -m venv vit

#this will create a folder named vit in root directory
# activate this env by 
source vit/bin/activate
```

virtual env creation - `Windows`
```
# in root directory
pip install virtualenv
virtualenv vit

#this will create a folder named vit in root directory
# activate this env by
vit\Scripts\activate
```

install all requirements
```
pip install --upgrade pip
pip install --upgrade TensorFlow
pip install validators
pip install matplotlib
pip install numpy
pip install opencv-python
```

### classification

Repo's root directory contains default sample (`sample.jpg`) for running classification. 

For running object detection run this line of code

load default sample
```
python3 run.py
```
output : `cabbage butterfly`

give image path as argument to load image of your choice
```
python3 run.py --image sample.jpg
```
give url as argument to load image from web
```
python3 run.py --image image_url
```

## References
- [vit-keras]([https://github.com/Leonardo-Blanger/detr_tensorflow](https://github.com/faustomorales/vit-keras): reference repo
- [Attention Is All You Need paper](https://arxiv.org/pdf/1706.03762.pdf)

## Acknowledgement
1. [Mr. Thomas Paul](https://github.com/mrtpk)
2. [Mr. Sambhu Surya Mohan](https://github.com/sambhusuryamohan)
