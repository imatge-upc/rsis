# Recurrent Neural Networks for Semantic Instance Segmentation

Code supporting the paper:

```
whatever 
```

If you find it useful, please consider citing !

## Installation
- Clone the repo:

```shell
git clone https://github.com/imatge-upc/rsis.git
```

- Install requirements ```pip install -r requirements.txt``` 
- Install [PyTorch 0.2](http://pytorch.org/) (choose the whl file according to your setup):

```shell
pip install http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp27-cp27mu-manylinux1_x86_64.whl  
pip install torchvision
```

- Compile COCO Python API and add it to your ```PYTHONPATH```:

```shell
cd src/coco/PythonAPI;
make
```

```shell
# Run from the root directory of this project
export PYTHONPATH=$PYTHONPATH:./src/coco/PythonAPI
```

## Data

### Pascal VOC 2012

- Download Pascal VOC 2012:

```shell
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xvf VOCtrainval_11-May-2012.tar
```
- Download additional VOC annotations from Hariharan et al. [Semantic contours from inverse detectors.](http://home.bharathh.info/pubs/pdfs/BharathICCV2011.pdf) ICCV 2011:

```shell
# berkeley augmented Pascal VOC
wget http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz # 1.3 GB
tar zxvf benchmark.tgz
```

- Create a merged dataset out of the two sets of images and annotations: 
```
python src/dataloader/pascalplus_gen.py --voc_dir /path/to/pascal --contours_dir /path/to/additional/dataset --vocplus_dir /path/to/merged
```
- Precompute instance and semantic segmentation masks & ground truth files in COCO format:

```
python src/dataloader/pascal_precompute.py --make_cocofile --split train --pascal_dir /path/to/merged
``` 

You must run this three times for the different splits (train, val and test).

### CVPPP

TODO: Add instructions

### Cityscapes

TODO: Add instructions

## Training

- Train the model with ```python train.py -model_name model_name```. Checkpoints and logs will be saved under ```../models/model_name```. Other arguments can be passed as well.
- For convenience, scripts to train with typical parameters are provided under ```scripts/```.
- Visdom can be enabled to monitor training losses and outputs:
	- First run the visdom server with```python -m visdom.server```.
	- Run training with the ```--visdom``` flag. Navigate to ```localhost:8097``` to visualize training curves.
- Plot loss curves at any time with ```python plot_curves.py -model_name model_name```. The same script works for plotting semantic segmentation loss curves if run with the flag ```--use_ss_model```.
- You can resume training a model with the flag --resume:

```
python train.py -model_name model_name -epoch_resume 5
```

Use ```-epoch_resume``` to specify the epoch in which training was stopped. This is useful if some changes happen after a certain number of epochs (eg fine tuning the base model).

## Evaluation

Run ```python eval.py -model_name model_name -eval_split test``` to compute AP figures. The file will be saved in the model's folder. If you want to save figures with results, run the command with the ```--display```flag.

There's also a ```test_ss.py``` script to visualize outputs of the semantic segmentation model trained with ```train_ss.py```.


## Additional notes to GPI users	

Helpful commands to train on the GPI cluster and get visualizations in your browser:

- Start server: with ```srun --tunnel $UID:$UID python -m visdom.server -port $UID```. 
- Check the node where the server launched and (eg. ```c3```).
- Run training with: 
```
srun --gres=gpu:1,gmem:12G --mem=10G python train.py --visdom -port $UID -server http://c3
``` 
  Notice that the port and the server must match the ones used in the previous run.
- ```echo $UID``` to know which port you are using.
- ssh tunnel (run this in local machine): ```ssh -L 8889:localhost:YOUR_UID -p2222 user@imatge.upc.edu```.
- Navigate to ```localhost:8889``` in your browser locally.

## Contact

For questions and suggestions use the issues section or send an e-mail to amaia.salvador@upc.edu

