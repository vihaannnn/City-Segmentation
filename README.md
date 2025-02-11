# Seg-Fil

## An image segmentation and replacement model for landscapes
Sef-Fil is an image segmentation model that allows you to remove specific classes within an image and then replace them with diffusion based in fill. 

## Dataset
The data used to train this model can be found [here](https://www.coursera.org/learn/convolutional-neural-networks). A more detailed discussion of its limitations and potential pitfalls can be found in the [Ethics statement](Ethics_statement.md).

## Naïve model
in the [Naïve model](Naïve_CV.py) there are very simple functions for converting to greyscale, segmenting by a set amount and then reconstructing the image. An example of usage is given in the 'Example use' function.

## Classical model

In the [Classical model](Class_ML_CV.py) there is extensive functions for testing and evaluation of the segmentation technique. A demonstration of these evaluation techniques is shown in the eval_on_dataset function. The segment_Kmeans function provides the core functionality for the segmentation, it should be passed an image and returns the segmented version.