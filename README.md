# luojia1-cloud-detection
Detect cloud in nighttime images from Luojia-1 satellite using basic filters.
![demo](https://raw.githubusercontent.com/DEDZTBH/luojia1-cloud-detection/master/demo.gif)

## Algorithm
Unsharp mask is usually used to improve clarity of an image. When I apply it to the nighttime satellite image, I found that streetlights are brought out by the filter while the cloud stays pretty much the same. 

Then removes all pixels over a threshold (image is chunked and each chunk have different threshold). The image is now only left with clouds and dimmer parts of the streetlights. 

Then average blurring which removes streetlights because their bright pixels are scarce (and very bright pixels are already removed) so they are averaged out.
 
Right now the cloud mask is taking its shape. Then some noise reduction by remove dark pixels (different threshold for each chunk) and remove small connected components. 

The mask now predicts where there is certainly cloud, but there is most likely cloud near the predicted areas. To be more conservative with the prediction, I run a grey dilation and gaussian blur (we don't need it if we want binary mask) to extend the predicted area with cloud.

Here is a visualization of each step (binary mask) on an example image. Note that all images shown here are 5 times brighter.
![flowchart](https://raw.githubusercontent.com/DEDZTBH/luojia1-cloud-detection/master/flowchart.png)

Subtraction after each operation:
![flowchart](https://raw.githubusercontent.com/DEDZTBH/luojia1-cloud-detection/master/diff.png)

Addition after each operation:
![flowchart](https://raw.githubusercontent.com/DEDZTBH/luojia1-cloud-detection/master/diff2.png)

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

- hi (in remove_bright_single): The threshold which all pixels with value over it is removed in step remove_bright.
- ksize (in average_blur): The kernel size used for step average_blur. It should be odd, and higher value remove streetlight better but also is more likely to remove cloud.
- lo (in remove_dark_single): The threshold which all pixels with value below it is removed in step remove_dark.
- obj_threshold (in remove_small_obj): The threshold which all connected components smaller with pixel number small than it is removed. Higher value remove small bright noises but is also more likely to remove small cloud.
- gdsize (in grey_dilation): The size of grey dilation. Use larger value for more conservative prediction. Also, since there is no gaussian_blur for binary mask, this should be set a larger value for binary mask prediction.
- gksize, gstd (in gaussian_blur): The kernel size (odd) and std for gaussian_blur.