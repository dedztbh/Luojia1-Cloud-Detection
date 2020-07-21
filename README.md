# luojia1-cloud-detection
Detect cloud in nighttime images from Luojia-1 satellite using basic filters.

## Algorithm
Unsharp mask is usually used to improve clarity of an image. When I apply it to the nighttime satellite image, I found that streetlights are brought out by the filter while the cloud stays pretty much the same. 

Then lowpass filter that removes all pixels over a threshold. The image is now only left with clouds and dimmer parts of the streetlights. 

Then average blurring which removes streetlights because their bright pixels are scarce (and very bright pixels are already removed) so they are averaged out.
 
Right now the cloud mask is taking its shape. Then are some noise reduction with high pass filter and remove small connected components. 

The mask now predicts where there is certainly cloud, but there is most likely cloud near the predicted areas. To be more conservative with the prediction, I run a grey dilation and gaussian blur to extend the predicted area with cloud.

Here is a visualization of each step on an example image. Note that all images shown here are 5 times brighter.
![flowchart](https://github.com/DEDZTBH/luojia1-cloud-detection/master/flowchart.png)

For implementation, usage, and more examples, please refer to main.ipynb and core.py

## Data
All Luojia-1 satellite data is found here: http://59.175.109.173:8888/

The image downloaded are stored in a stretched int32 format that can be converted back to float point with formula L = DN^(3/2)*10^-10 where DN is the digital number and L is the actual radiance in W/(m^2 * sr * μm). My current set of parameters works with 10^5 L.

## Parameters
Yes, these are mostly hard coded for now. I am trying to see if it is possible to use different thresholds for individual images. 