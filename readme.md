# Use Hover-net to train your own model

This post will tell you how to train a HoVer-Net using your own datasets and to process image tiles or whole-slide images (WSIs). HoVer-Net is a multiple branch network that performs nuclear instance segmentation and classification within a single network. The network leverages the horizontal and vertical distances of nuclear pixels to their centers of mass to separate clustered cells. A dedicated up-sampling branch is used to classify the nuclear type for each segmented instance. First of first, you'd better refer to the official PyTorch implementation of [HoVer-Net](https://github.com/vqdang/hover_net).

## 1. Environment requirements
I use PyTorch version 1.8 with CUDA 10.2. You can refer to the official PyTorch implementation:
```
conda env create -f environment.yml
conda activate hovernet
pip install torch==1.6.0 torchvision==0.7.0
```
## 2. Some important scripts
Below are the some scripts that you'd better know at first:

1. ```config.py``` : main configuration file,  
 - model mode, number of nuclei types (N+1, 1 for background), training data directory, log directory, network input and output size, etc, can be set in ```config.py```. ```config.py``` will call another configuration file in ```./models/hovernet/opt.py```
  - Note, Hover-net provides two model mode. 'original' model mode refers to the method described in the original medical image analysis paper with a 270x270 patch input and 80x80 patch output. 'fast' model mode uses a 256x256 patch input and 164x164 patch output. Also, if using a model trained only for segmentation, nr_types must be set to 0.
2. ```./models/hovernet/opt.py``` : model configuration file. Hover-net adopts a two-stage training strategy. So, in "phase_list", there are two "run_info" dicts, each saves model parameters for one stage. 
  - In the first stage, the encoder of Hover-net will be initialized with [pretrained Preact-ResNet50 weights](https://drive.google.com/file/d/1KntZge40tAHgyXmHYVqZZ5d2p_4Qr2l5/view) on the ImageNet dataset, the encoder is fixed and only train the three decoders for 50 epochs.
  - In the second stage, both encoder and three decoders will be trained for another 50 epochs.
  - Optimizer, learning rate, loss functions, batch size, training epochs, number of threads for dataloader, pre-trained model directory can be set in ```opt.py```.
3. ```dataset.py```: defines the dataset classes. Load the original patch-level RGB images and its annotations. You can define your own dataset in this file.
4. ```GenerateMasks.py``` :  Convert the manually annotated nucleus contours into a mask. Parse the ```.session.xml``` file of Sedeen Viewer, convert it into a mask, and save it as a dictionary file with keys: inst_map (for nuclei instance) and inst_type (for nuclei type).
5. ```extract_patches.py```: extracts patches from original images. This script is used to make training and validation datasets. It takes patch-level images (e.g, 1000x1000 pixel) as input and crops them into smaller tiles (e.g, with size 540x540 pixel^2), and zip the cropped tile together with its annotations as a numpy array. 
  - For the numpy array, the first three channels save the image, and the rest several channel save its annotations.
6. ```convert_format.py``` : Used to convert output (.json) to a format that can be used for visualisaton with QuPath. Note, this is only used for tile segmentation results; not WSI.
7. ```type_info.json``` :  is used to specify what RGB colours are used in the overlay. Make sure to modify this for different datasets and if you would like to generally control overlay boundary colours.
8. ```compute_stats.py```:  main metric computation script
9. ```run_train.py```:  main training script
10. ```run_infer.py```:  main inference script for tile and WSI processing
11.  ```plot_heat_map.py```:  According to the inference results, parse the generated ```.json``` file and count the number of different types of nuclei. Draw a density map for WSIs and do hot spot analysis.
12. ```convert_chkpt_tf2pytorch```: convert tensorflow .npz model trained in original repository to pytorch supported .tar format.
13. ```run_tile.sh``` :  Script for inference on patch-level images.
14. ```run_wsi.sh``` :  Script for inference on WSI.
## 3. Other main directories:
1. ```dataloader/```: the data loader and augmentation pipeline
2. ```metrics/```: scripts for metric calculation
3. ```misc/```: utils that are
4. ```models/```: model definition, along with the main run step and hyperparameter settings
5. ```run_utils/```: defines the train/validation loop and callbacks
## 4. How to train your own model？
The simple answer is we need three steps. First, prepare your training data. Then, train the model. Last, test the model with your test data. Next, I will take the instance segmenttaion in IHC-stained images as an example. 

### 4.1 Prepare your own training data
1. Crop predefined size patches from WSIs, e.g, the patch size is 1000x1000 pixels.
2. Make annotations for patches. Suppose you use Sedeen Viewer to manually annotate nuclei contour. Then you'll use the Polygon tool to draw the nuclei contour, and use different colors to distinguish nuclei types. Finally, Sedeen Viewer will generate a ```session.xml``` file.
3. Use ```GenerateMasks.py``` to parse the ```session.xml``` file and convert the contour annotations into pixel-level annotations (mask). Then make the annotation as a dict, with two keys as following. And last, save it as ```.mat``` file. 
  - 'inst_map': instance map containing values from 0 to N, where N is the number of nuclei;
  - 'inst_type': list of length N containing predictions for each nucleus.
### 4.2 Trian the model
#### Before traning
If your patches is bigger than 540 x 540 pixels, it must be extracted using ```extract_patches.py```. 
- Set the input and output file addresses in ```extract_patches.py```. Remember, you need to call your ```dataset.py``` for image and annotation load.
- For instance segmentation, patches are stored as a 4 dimensional numpy array with channels [RGB, inst_map]. Here, inst_map is the instance segmentation ground truth. I.e pixels range from 0 to N, where 0 is background and N is the number of nuclear instances for that particular image.
- For simultaneous instance segmentation and classification, patches are stored as a 5 dimensional numpy array with channels [RGB, inst_map, inst_type]. Here, inst_type is the ground truth of the nuclear type. I.e every pixel ranges from 0-K, where 0 is background and K is the number of classes.

Set paths and traning hyperparameters.
  - Set nr_type, model mode and path to the data directories in ```config.py```
  - Set path where checkpoints will be saved in ```config.py```
  - Set path to [pretrained Preact-ResNet50 weights](https://drive.google.com/file/d/1KntZge40tAHgyXmHYVqZZ5d2p_4Qr2l5/view) in ```models/hovernet/opt.py```. 
  -  Modify hyperparameters, including number of epochs and learning rate in ```models/hovernet/opt.py```.
#### Traning
To visualise the training dataset as a sanity check before training use:

  ```python run_train.py --view='train'```

To initialise the training script with GPUs 0 and 1, the command is:

  ```python run_train.py --gpu='0,1' ```

### 4.3 Inference
Hover-Net provides two inference modes, on patch-level images and WSIs. 
#### Data Format
- Input:
  - Standard images files, including``` .png```, ```.jpg``` and ```.tiff```.
  - WSIs supported by ```OpenSlide```, including ```.svs```, ```.tif```, ```.ndpi``` and ```.mrxs```.
- Output:
  - Both image tiles and whole-slide images output a ```.json``` file with keys:
    - 'bbox': bounding box coordinates for each nucleus
    - 'centroid': centroid coordinates for each nucleus
    - 'contour': contour coordinates for each nucleus
    - 'type_prob': per class probabilities for each nucleus (default configuration doesn't output this)
    - 'type': prediction of category for each nucleus

  - Image tiles output a ```.mat``` file, with keys:
    - 'raw': raw output of network (default configuration doesn't output this)
    - 'inst_map': instance map containing values from 0 to N, where N is the number of nuclei
    - 'inst_type': list of length N containing predictions for each nucleus
  - Image tiles output a png overlay of nuclear boundaries on top of original RGB image
##### Test on patch-level images
Usage: you can set parameters in ```run_tile.sh```.
```
  chmod +x run_tile.sh
  ./run_tile.sh
```
Options 
```
  -h --help                   Show this string.
  --version                   Show version.
  --gpu=<id>                  GPU list. [default: 0]
  --nr_types=<n>              Number of nuclei types to predict. [default: 0]
  --type_info_path=<path>     Path to a json define mapping between type id, type name, 
                              and expected overlay color. [default: '']
  --model_path=<path>         Path to saved checkpoint.
  --model_mode=<mode>         Original HoVer-Net or the reduced version used in PanNuke / MoNuSAC, 'original' or 'fast'. [default: fast]
  --nr_inference_workers=<n>  Number of workers during inference. [default: 8]
  --nr_post_proc_workers=<n>  Number of workers during post-processing. [default: 16]
  --batch_size=<n>            Batch size. [default: 128]
```
##### Test on WSIs
For WSIs, it may take 2 hours when using 4 GPUs
Usage: you can set parameters in ```run_wsi.sh```.
```
  chmod +x run_wsi.sh
  ./run_wsi.sh
```
Options: 
```
    --input_dir=<path>      Path to input data directory. Assumes the files are not nested within directory.
    --output_dir=<path>     Path to output directory.
    --cache_path=<path>     Path for cache. Should be placed on SSD with at least 100GB. [default: cache]
    --mask_dir=<path>       Path to directory containing tissue masks. 
                            Should have the same name as corresponding WSIs. [default: '']

    --proc_mag=<n>          Magnification level (objective power) used for WSI processing. [default: 40]
    --ambiguous_size=<int>  Define ambiguous region along tiling grid to perform re-post processing. [default: 128]
    --chunk_shape=<n>       Shape of chunk for processing. [default: 10000]
    --tile_shape=<n>        Shape of tiles for processing. [default: 2048]
    --save_thumb            To save thumb. [default: False]
    --save_mask             To save mask. [default: False]
```
## 5. Hotspot analysis on WSIs
We utilize heatmaps to visualize the density of the positive nucleus in the WSIs. We also apply hotspot analysis to find the top k most positive areas in each WSIs.
### 5.1 plot the density map for WSIs
In ```plot_heat_map.py```, the function ```plot_density_map()``` is used to plot the density map of positive nuclei.
  - First, we split each WSI into ```N x N``` (e.g, 50 x 50) pixel^2 tiles， and count the number of nuclei centers of each type contained within each tile. 
  - Then, calculate the average positive nuclei rate per pixel within each tile, e.g, the number of positive nuclei in the tile is ```n```, then ```the average positive nuclei rate per pixel = n / NxN ```.
  - Last, draw the normalized positive nuclei rate by heatmap.
### 5.2 Find the top k hotspots
In ```plot_heat_map.py```, the function ```most_positive_cord_center()``` is used to find the top k most positive areas from the WSIs automatically.
  - Note, ```most_positive_cord_center()``` return bounding box list. 
  - When enumerating bounding boxes, bounding boxes that overlap with the already selected bounding boxes will be discarded. 
  - ```create_xml()``` in ```plot_heat_map.py``` converts the bounding box to a ``.xml`` file that Sedeen Viewer can open directly.