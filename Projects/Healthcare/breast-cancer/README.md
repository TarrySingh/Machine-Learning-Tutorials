# Deep Neural Networks Improve Radiologists' Performance in Breast Cancer Screening

## Introduction
This is an implementation of the model used for breast cancer classification as described in our paper [Deep Neural Networks Improve Radiologists' Performance in Breast Cancer Screening](https://arxiv.org/abs/1903.08297). The implementation allows users to get breast cancer predictions by applying one of our pretrained models: a model which takes images as input (*image-only*) and a model which takes images and heatmaps as input (*image-and-heatmaps*). 

* Input images: 2 CC view mammography images of size 2677x1942 and 2 MLO view mammography images of size 2974x1748. Each image is saved as 16-bit png file and gets standardized separately before being fed to the models.
* Input heatmaps: output of the patch classifier constructed to be the same size as its corresponding mammogram. Two heatmaps are generated for each mammogram, one for benign and one for malignant category. The value of each pixel in both of them is between 0 and 1.
* Output: 2 predictions for each breast, probability of benign and malignant findings: `left_benign`, `right_benign`, `left_malignant`, and `right_malignant`.

Both models act on screening mammography exams with four standard views (L-CC, R-CC, L-MLO, R-MLO). As a part of this repository, we provide 4 sample exams (in `sample_data/images` directory and exam list stored in `sample_data/exam_list_before_cropping.pkl`). Heatmap generation model and cancer classification models are implemented in PyTorch. 

## Prerequisites

* Python (3.6)
* PyTorch (0.4.0)
* torchvision (0.2.0)
* NumPy (1.14.3)
* SciPy (1.0.0)
* H5py (2.7.1)
* imageio (2.4.1)
* pandas (0.22.0)
* tqdm (4.19.8)
* opencv-python (3.4.2)
* tensorflow (1.14.0)

## License

This repository is licensed under the terms of the GNU AGPLv3 license.

## How to run the code

`run.sh` will automatically run the entire pipeline and save the prediction results in csv. 

We recommend running the code with a gpu (set by default). To run the code with cpu only, please change `DEVICE_TYPE` in run.sh to 'cpu'.  

If running the individual Python scripts, please include the path to this repository in your `PYTHONPATH` . 

You should obtain the following outputs for the sample exams provided in the repository. 

Predictions using *image-only* model (found in `sample_output/image_predictions.csv` by default):

| index | left_benign | right_benign | left_malignant | right_malignant |
| ----- | ----------- | ------------ | -------------- | --------------- |
| 0     | 0.0580      | 0.0091       | 0.0754         | 0.0179          |
| 1     | 0.0646      | 0.0012       | 0.9536         | 0.7258          |
| 2     | 0.4388      | 0.2325       | 0.3526         | 0.1061          |
| 3     | 0.3765      | 0.0909       | 0.6483         | 0.2579          |


Predictions using *image-and-heatmaps* model (found in `sample_output/imageheatmap_predictions.csv` by default):

| index | left_benign  | right_benign | left_malignant | right_malignant |
| ----- | ------------ | ------------ | -------------- | --------------- |
| 0     | 0.0612       | 0.0099       | 0.0754         | 0.0179          |
| 1     | 0.0507       | 0.0009       | 0.8025         | 0.9000          |
| 2     | 0.2877       | 0.2524       | 0.2286         | 0.0461          |
| 3     | 0.4181       | 0.3174       | 0.3172         | 0.0485          |



## Data

To use one of the pretrained models, the input is required to consist of at least four images, at least one for each view (L-CC, L-MLO, R-CC, R-MLO). 

The original 12-bit mammograms are saved as rescaled 16-bit images to preserve the granularity of the pixel intensities, while still being correctly displayed in image viewers.

`sample_data/exam_list_before_cropping.pkl` contains a list of exam information before preprocessing. Each exam is represented as a dictionary with the following format:

```python
{
  'horizontal_flip': 'NO',
  'L-CC': ['0_L_CC'],
  'R-CC': ['0_R_CC'],
  'L-MLO': ['0_L_MLO'],
  'R-MLO': ['0_R_MLO'],
}
```

We expect images from `L-CC` and `L-MLO` views to be facing right direction, and images from `R-CC` and `R-MLO` views are facing left direction. `horizontal_flip` indicates whether all images in the exam are flipped horizontally from expected. Values for `L-CC`, `R-CC`, `L-MLO`, and `R-MLO` are list of image filenames without extension and directory name. 

Additional information for each image gets included as a dictionary. Such dictionary has all 4 views as keys, and the values are the additional information for the corresponding key. For example, `window_location`, which indicates the top, bottom, left and right edges of cropping window, is a dictionary that has 4 keys and has 4 lists as values which contain the corresponding information for the images. Additionally, `rightmost_pixels`, `bottommost_pixels`, `distance_from_starting_side` and `best_center` are added after preprocessing. 
Description for these attributes can be found in the preprocessing section. 
The following is an example of exam information after cropping and extracting optimal centers:

```python
{
  'horizontal_flip': 'NO',
  'L-CC': ['0_L_CC'],
  'R-CC': ['0_R_CC'],
  'L-MLO': ['0_L_MLO'],
  'R-MLO': ['0_R_MLO'],
  'window_location': {
    'L-CC': [(353, 4009, 0, 2440)],
    'R-CC': [(71, 3771, 952, 3328)],
    'L-MLO': [(0, 3818, 0, 2607)],
    'R-MLO': [(0, 3724, 848, 3328)]
   },
  'rightmost_points': {
    'L-CC': [((1879, 1958), 2389)],
    'R-CC': [((2207, 2287), 2326)],
    'L-MLO': [((2493, 2548), 2556)],
    'R-MLO': [((2492, 2523), 2430)]
   },
  'bottommost_points': {
    'L-CC': [(3605, (100, 100))],
    'R-CC': [(3649, (101, 106))],
    'L-MLO': [(3767, (1456, 1524))],
    'R-MLO': [(3673, (1164, 1184))]
   },
  'distance_from_starting_side': {
    'L-CC': [0],
    'R-CC': [0],
    'L-MLO': [0],
    'R-MLO': [0]
   },
  'best_center': {
    'L-CC': [(1850, 1417)],
    'R-CC': [(2173, 1354)],
    'L-MLO': [(2279, 1681)],
    'R-MLO': [(2185, 1555)]
   }
}
```

The labels for the included exams are as follows:

| index | left_benign | right_benign | left_malignant | right_malignant |
| ----- | ----------- | ------------ | -------------- | --------------- |
| 0     | 0           | 0            | 0              | 0               |
| 1     | 0           | 0            | 0              | 1               |
| 2     | 1           | 0            | 0              | 0               |
| 3     | 1           | 1            | 1              | 1               |


## Pipeline

The pipeline consists of four stages.

1. Crop mammograms
2. Calculate optimal centers
3. Generate Heatmaps
4. Run classifiers

The following variables defined in `run.sh` can be modified as needed:
* `NUM_PROCESSES`: The number of processes to be used in preprocessing (`src/cropping/crop_mammogram.py` and `src/optimal_centers/get_optimal_centers.py`). Default: 10.
* `DEVICE_TYPE`: Device type to use in heatmap generation and classifiers, either 'cpu' or 'gpu'. Default: 'gpu'
* `NUM_EPOCHS`: The number of epochs to be averaged in the output of the classifiers. Default: 10.
* `HEATMAP_BATCH_SIZE`: The batch size to use in heatmap generation. Default: 100.
* `GPU_NUMBER`: Specify which one of the GPUs to use when multiple GPUs are available. Default: 0. 

* `DATA_FOLDER`: The directory where the mammogram is stored.
* `INITIAL_EXAM_LIST_PATH`: The path where the initial exam list without any metadata is stored.
* `PATCH_MODEL_PATH`: The path where the saved weights for the patch classifier is saved.
* `IMAGE_MODEL_PATH`: The path where the saved weights for the *image-only* model is saved.
* `IMAGEHEATMAPS_MODEL_PATH`: The path where the saved weights for the *image-and-heatmaps* model is saved.

* `CROPPED_IMAGE_PATH`: The directory to save cropped mammograms.
* `CROPPED_EXAM_LIST_PATH`: The path to save the new exam list with cropping metadata.
* `EXAM_LIST_PATH`: The path to save the new exam list with best center metadata.
* `HEATMAPS_PATH`: The directory to save heatmaps.
* `IMAGE_PREDICTIONS_PATH`: The path to save predictions of *image-only* model.
* `IMAGEHEATMAPS_PREDICTIONS_PATH`: The path to save predictions of *image-and-heatmaps* model.


### Preprocessing

Run the following commands to crop mammograms and calculate information about augmentation windows.

#### Crop mammograms
```bash
python3 src/cropping/crop_mammogram.py \
    --input-data-folder $DATA_FOLDER \
    --output-data-folder $CROPPED_IMAGE_PATH \
    --exam-list-path $INITIAL_EXAM_LIST_PATH  \
    --cropped-exam-list-path $CROPPED_EXAM_LIST_PATH  \
    --num-processes $NUM_PROCESSES
```
`src/import_data/crop_mammogram.py` crops the mammogram around the breast and discards the background in order to improve image loading time and time to run segmentation algorithm and saves each cropped image to `$PATH_TO_SAVE_CROPPED_IMAGES/short_file_path.png` using h5py. In addition, it adds additional information for each image and creates a new image list to `$CROPPED_IMAGE_LIST_PATH` while discarding images which it fails to crop. Optional --verbose argument prints out information about each image. The additional information includes the following:
- `window_location`: location of cropping window w.r.t. original dicom image so that segmentation map can be cropped in the same way for training.
- `rightmost_points`: rightmost nonzero pixels after correctly being flipped.
- `bottommost_points`: bottommost nonzero pixels after correctly being flipped.
- `distance_from_starting_side`: records if zero-value gap between the edge of the image and the breast is found in the side where the breast starts to appear and thus should have been no gap. Depending on the dataset, this value can be used to determine wrong value of `horizontal_flip`.


#### Calculate optimal centers
```bash
python3 src/optimal_centers/get_optimal_centers.py \
    --cropped-exam-list-path $CROPPED_EXAM_LIST_PATH \
    --data-prefix $CROPPED_IMAGE_PATH \
    --output-exam-list-path $EXAM_LIST_PATH \
    --num-processes $NUM_PROCESSES
```
`src/optimal_centers/get_optimal_centers.py` outputs new exam list with additional metadata to `$EXAM_LIST_PATH`. The additional information includes the following:
- `best_center`: optimal center point of the window for each image. The augmentation windows drawn with `best_center` as exact center point could go outside the boundary of the image. This usually happens when the cropped image is smaller than the window size. In this case, we pad the image and shift the window to be inside the padded image in augmentation. Refer to [the data report](https://cs.nyu.edu/~kgeras/reports/datav1.0.pdf) for more details.


### Heatmap Generation
```bash
python3 src/heatmaps/run_producer.py \
    --model-path $PATCH_MODEL_PATH \
    --data-path $EXAM_LIST_PATH \
    --image-path $CROPPED_IMAGE_PATH \
    --batch-size $HEATMAP_BATCH_SIZE \
    --output-heatmap-path $HEATMAPS_PATH \
    --device-type $DEVICE_TYPE \
    --gpu-number $GPU_NUMBER
```

`src/heatmaps/run_producer.py` generates heatmaps by combining predictions for patches of images and saves them as hdf5 format in `$HEATMAPS_PATH` using `$DEVICE_TYPE` device. `$DEVICE_TYPE` can either be 'gpu' or 'cpu'. `$HEATMAP_BATCH_SIZE` should be adjusted depending on available memory size.  An optional argument `--gpu-number`  can be used to specify which GPU to use.

### Running the models

`src/modeling/run_model.py` can provide predictions using cropped images either with or without heatmaps. When using heatmaps, please use the`--use-heatmaps` flag and provide appropriate the `--model-path` and `--heatmaps-path` arguments. Depending on the available memory, the optional argument `--batch-size` can be provided. Another optional argument `--gpu-number` can be used to specify which GPU to use.

#### Run image only model
```bash
python3 src/modeling/run_model.py \
    --model-path $IMAGE_MODEL_PATH \
    --data-path $EXAM_LIST_PATH \
    --image-path $CROPPED_IMAGE_PATH \
    --output-path $IMAGE_PREDICTIONS_PATH \
    --use-augmentation \
    --num-epochs $NUM_EPOCHS \
    --device-type $DEVICE_TYPE \
    --gpu-number $GPU_NUMBER
```

This command makes predictions only using images for `$NUM_EPOCHS` epochs with random augmentation and outputs averaged predictions per exam to `$IMAGE_PREDICTIONS_PATH`. 

#### Run image+heatmaps model 
```bash
python3 src/modeling/run_model.py \
    --model-path $IMAGEHEATMAPS_MODEL_PATH \
    --data-path $EXAM_LIST_PATH \
    --image-path $CROPPED_IMAGE_PATH \
    --output-path $IMAGEHEATMAPS_PREDICTIONS_PATH \
    --use-heatmaps \
    --heatmaps-path $HEATMAPS_PATH \
    --use-augmentation \
    --num-epochs $NUM_EPOCHS \
    --device-type $DEVICE_TYPE \
    --gpu-number $GPU_NUMBER
```

This command makes predictions using images and heatmaps for `$NUM_EPOCHS` epochs with random augmentation and outputs averaged predictions per exam to `$IMAGEHEATMAPS_PREDICTIONS_PATH`. 

## Reference

If you found this code useful, please cite our paper:

**Deep Neural Networks Improve Radiologists' Performance in Breast Cancer Screening**\
Nan Wu, Jason Phang, Jungkyu Park, Yiqiu Shen, Zhe Huang, Masha Zorin, Stanisław Jastrzębski, Thibault Févry, Joe Katsnelson, Eric Kim, Stacey Wolfson, Ujas Parikh, Sushma Gaddam, Leng Leng Young Lin, Kara Ho, Joshua D. Weinstein, Beatriu Reig, Yiming Gao, Hildegard Toth, Kristine Pysarenko, Alana Lewin, Jiyon Lee, Krystal Airola, Eralda Mema, Stephanie Chung, Esther Hwang, Naziya Samreen, S. Gene Kim, Laura Heacock, Linda Moy, Kyunghyun Cho, Krzysztof J. Geras\
2019

    @article{wu2019breastcancer, 
        title = {Deep Neural Networks Improve Radiologists' Performance in Breast Cancer Screening},
        author = {Nan Wu, Jason Phang, Jungkyu Park, Yiqiu Shen, Zhe Huang, Masha Zorin, Stanis\l{}aw Jastrz\k{e}bski, Thibault F\'{e}vry, Joe Katsnelson, Eric Kim, Stacey Wolfson, Ujas Parikh, Sushma Gaddam, Leng Leng Young Lin, Kara Ho, Joshua D. Weinstein, Beatriu Reig, Yiming Gao, Hildegard Toth, Kristine Pysarenko, Alana Lewin, Jiyon Lee, Krystal Airola, Eralda Mema, Stephanie Chung, Esther Hwang, Naziya Samreen, S. Gene Kim, Laura Heacock, Linda Moy, Kyunghyun Cho, Krzysztof J. Geras}, 
        journal = {arXiv:1903.08297},
        year = {2019}
    }
