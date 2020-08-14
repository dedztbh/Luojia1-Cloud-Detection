# Luojia-1 Satellite Visible Band Nighttime Imagery Cloud Detection 
Detect cloud in nighttime images from Luojia-1 satellite using basic filters and image processing techniques.

This demo shows the algorithm running on images from Luojia-1 satellite as it flew through an area. The red is sensor reading and the green is cloud mask prediction.
![demo](https://raw.githubusercontent.com/DEDZTBH/luojia1-cloud-detection/master/demo_img/demo.gif)

## Table of Contents
- [Luojia-1 Satellite Visible Band Nighttime Imagery Cloud Detection](#luojia-1-satellite-visible-band-nighttime-imagery-cloud-detection)
  * [Table of Contents](#table-of-contents)
  * [Algorithm](#algorithm)
    + [Overview](#overview)
    + [Description](#description)
  * [Results](#results)
  * [Files](#files)
  * [Data](#data)
  * [Parameters](#parameters)

## Algorithm
### Overview
![system_overview](https://raw.githubusercontent.com/DEDZTBH/luojia1-cloud-detection/master/demo_img/system_overview.png)

### Description
Unsharp mask is a high-pass filter usually used to improve clarity of an image. When applied to the nighttime satellite image from the Luojia-1 satellite, the scatter of streetlights in cloudy areas is greatly reduced while the cloud stays mostly intact. This makes it easier to separate streetlights from clouds.

The next step is to remove all pixels brighter than a threshold. Due to many images have very different local properties (eg. urban vs rural, cloudy vs clear), the threshold is computed and applied to individual chunks of the image. This step is able to eliminate the brighter part of streetlights. The image is now only left with clouds and dimmer parts of the streetlights. 

The next step is average blurring, a low-pass filter, which removes streetlights with no nearby cloud because of their relatively bright and sparse pixels (very bright pixels are already removed in the previous step).
 
Right now the cloud mask is starting to take shape, but some noise reduction is still needed. The next step is to remove all pixels darker than a threshold. Similarly, the threshold is computed and applied to individual chunks of the image.

If the desired output is a binary mask instead of a gradient one, convert the image into binary.

There are some remaining spotty bright noises that cannot be removed by the previous steps. The next step is to generate statistics of all connected components in the mask and remove the ones with area smaller than a threshold (eg. the mean size of all connected components). 

The mask now predicts where there are certainly clouds, but there are most likely clouds near the predicted areas. To be more conservative with the prediction, the next step is to run a grey dilation to extend the predicted area with cloud. If gradient prediction is needed, another gaussian blur can be applied to smooth out the dilated mask.

Here is a visualization of each step (binary mask) on an example image. Note that all images shown here are 5 times brighter.
![flowchart](https://raw.githubusercontent.com/DEDZTBH/luojia1-cloud-detection/master/demo_img/flowchart.png)

Subtraction after each operation:
![flowchart](https://raw.githubusercontent.com/DEDZTBH/luojia1-cloud-detection/master/demo_img/diff.png)

Addition after each operation:
![flowchart](https://raw.githubusercontent.com/DEDZTBH/luojia1-cloud-detection/master/demo_img/diff2.png)

Here is the full list of operations in core.py (binary mask):
```
cloud_mask_generate_procedure_binary = [
    unsharp,
    remove_bright,
    average_blur,
    remove_dark,
    to_binary,  # if we want binary mask
    remove_small_obj,
    grey_dilation,
    # gaussian_blur  # if not binary mask
]
```
## Results
The algorithm generally is able to correctly predict most clouds, especially those that are lit by streetlights. The following 3 pairs of images are successful examples. The red is the original image and green is the predicted cloud mask.
![success1](https://raw.githubusercontent.com/DEDZTBH/luojia1-cloud-detection/master/demo_img/success1.png)
![success2](https://raw.githubusercontent.com/DEDZTBH/luojia1-cloud-detection/master/demo_img/success2.png)
![success3](https://raw.githubusercontent.com/DEDZTBH/luojia1-cloud-detection/master/demo_img/success3.png)

The algorithm does produce some false positive predictions, such as the below image, where there are large areas with dim streetlight. 
![failure1](https://raw.githubusercontent.com/DEDZTBH/luojia1-cloud-detection/master/demo_img/failure1.png)

Here is another failure example. The cloud at the bottom centre which is not well-lit is not being correctly recognized, and there are some false positive cases as well.
![failure2](https://raw.githubusercontent.com/DEDZTBH/luojia1-cloud-detection/master/demo_img/failure2.png)

## Files
- main.ipynp: Selected 12 images and shows them and their stats at each step. Also stitched images and their cloud mask together at last.
- batch.ipynp: Similar to main.ipynp but choosing random pictures
- core.py: Core implementation of algorithm
- flowchart_gen.py: Used to generate the diagram above
- image_tester.py: Generate cloud mask for specific image

## Data
All Luojia-1 satellite data is found here: http://59.175.109.173:8888/

The image downloaded are stored in a stretched int32 format that can be converted back to floating point with formula L = DN^(3/2)*10^-10 where DN is the digital number and L is the actual radiance in W/(m^2 * sr * μm). My current set of parameters works with 10^5 L.

## Parameters
Since images from other satellite have different properties, my current set of parameters might not work for other satellite images. Here are the list of parameters located in core.py which can be tuned:

- hi (in remove_bright_single): The threshold which all pixels brighter than it is removed in step remove_bright.
- remove_bright_window_size (in remove_bright): The size of the square used for chunking the image in step remove_bright.
- ksize (in average_blur): The kernel size used for step average_blur. It should be odd, and higher value remove streetlight better but also is more likely to remove cloud.
- lo (in remove_dark_single): The threshold which all pixels darker than it is removed in step remove_dark.
- binary_threshold (in to_binary): The threshold which all pixel darker than it is removed in step to_binary.
- remove_bright_window_size (in remove_bright): The size of the square used for chunking the image in step remove_dark.
- obj_threshold (in remove_small_obj): The threshold which all connected components with pixel number smaller than it is removed. Higher value remove small bright noises but is also more likely to remove small cloud.
- gdsize (in grey_dilation): The size of grey dilation. Use larger value for more conservative prediction. Also, since there is no gaussian_blur for binary mask, this should be set a larger value for binary mask prediction in general.
- gksize, gstd (in gaussian_blur): The kernel size (odd) and std for gaussian_blur.