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

Point ```args.pascal_dir``` to this folder.

### CVPPP

Download the training CVPPP dataset from their [website](https://www.plant-phenotyping.org/datasets-download). In our case we just worked with the A1 dataset. Extract the A1 package and point ```args.leaves_dir``` to this folder.  To obtain the test set for evaluation you will have to contact the organizers.

### Cityscapes

Download the Cityscapes dataset from their [website](https://www.cityscapes-dataset.com/downloads/). Extract the [images](https://www.cityscapes-dataset.com/file-handling/?packageID=3) and the [labels](https://www.cityscapes-dataset.com/file-handling/?packageID=1) into the same directory and point ```args.cityscapes_dir``` to it.


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

We provide bash scripts to display results and evaluate models for the three datasets. You can find them under the ```scripts``` folder.

In the case of cityscapes, the evaluation bash script will generate the results in the appropiate format to use the official evaluation [code](https://github.com/mcordts/cityscapesScripts). 

For CVPPP, the evaluation bash script will generate the results in the appropiate format to use the evaluation scripts that are provided with the [dataset](https://www.plant-phenotyping.org/datasets-download).

## Pretrained models

Download weights for models trained with:

- [Pascal VOC 2012](https://mega.nz/#!988QkDZS!3Mnn_A3XnhynEfsfPGKDUAPRmAMtFqyIf_0xrxU0obU)
- [Cityscapes](https://mega.nz/#!UhEESZ4a!UByeXh91wncbmJu-UaKJgpoZF5_KkuWEveTRxaKIxAE)
- [CVPPP](https://mega.nz/#!F5lBgJSD!DzOzaq6NBWPgLzVgPD1n9AmMmfNNmXLs0FguSUOhmO0)

Extract and place the obtained folder under ```models``` directory. 
You can then run ```eval.py``` with the downloaded model by setting ```args.model_name``` to the name of the folder.

## Contact

For questions and suggestions use the issues section or send an e-mail to amaia.salvador@upc.edu

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