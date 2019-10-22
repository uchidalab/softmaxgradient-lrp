# Softmax Gradient Layer-wise Relevance Propagation (SGLRP)

This is a Keras implementation of the the paper *Explaining Convolutional Neural Networks using Softmax Gradient Layer-wise Relevance Propagation* by Ryouhei Kuroki, Brian Kenji Iwana, and Seiichi Uchida. [arXiv](https://arxiv.org/abs/1908.04351)

## Abstract

Convolutional Neural Networks (CNN) have become state-of-the-art in the field of image classification. However, not everything is understood about their inner representations. This paper tackles the interpretability and explainability of the predictions of CNNs for multi-class classification problems. Specifically, we propose a novel visualization method of pixel-wise input attribution called Softmax-Gradient Layer-wise Relevance Propagation (SGLRP). The proposed model is a class discriminate extension to Deep Taylor Decomposition (DTD) using the gradient of softmax to back propagate the relevance of the output probability to the input image. Through qualitative and quantitative analysis, we demonstrate that SGLRP can successfully localize and attribute the regions on input images which contribute to a target object's classification. We show that the proposed method excels at discriminating the target objects class from the other possible objects in the images. We confirm that SGLRP performs better than existing Layer-wise Relevance Propagation (LRP) based methods and can help in the understanding of the decision process of CNNs. 

## Usage

### SGLRP Class

```
utils.visualizations.SGLRP(partial_model, target_id, relu=False, low=-1., high=1., **kwargs)
```
#### Arguments

**partial_model** : *Keras Model instance*

&nbsp;&nbsp;&nbsp;&nbsp;A keras model with the output of softmax cut off, but the input to the output layer intact. The partial model can be found using the innvestigate library using the ```innvestigate.utils.keras.graph.pre_softmax_tensors()``` function.
    
**target_id** : *int*

&nbsp;&nbsp;&nbsp;&nbsp;The index of the target class.
    
**relu** : *bool*

&nbsp;&nbsp;&nbsp;&nbsp;Controls if ReLU is applied to the visualization. ```True``` means that only the positive relevance is shown; ```False``` shows both the positive and negative relevance.
    
**low** : *float*

&nbsp;&nbsp;&nbsp;&nbsp;The upper bounds of the inputs.
    
**high** : *float*

&nbsp;&nbsp;&nbsp;&nbsp;The lower bounds of the inputs.
    
### SGLRP.analyze()
    
```
SGLRP.analyze(input_imgs)
```
#### Arguments

**input_imgs** : *4D array*

&nbsp;&nbsp;&nbsp;&nbsp;Array of input images in the format of ```(image_id, height, width, channel)```.
    

#### Returns

*4D array*

&nbsp;&nbsp;&nbsp;&nbsp;Array of heatmaps in the format of ```(image_id, height, width, channel)```.

### Simple Example

```
from utils.visualizations import SGLRP
import innvestigate.utils as iutils

model = [keras model]
partial_model = Model(inputs=model.inputs, outputs=iutils.keras.graph.pre_softmax_tensors(model.outputs)) 

target_id = [id of target class]
analysis = SGLRP(partial_model, target_id).analyze(input_imgs)

```

### Full Example

[Example Notebook](example.ipynb)

## Requires

This code was developed in Python 3.5.2.

```
pip install keras==2.2.4 numpy==1.14.5 matplotlib==2.2.2 innvestigate==1.0.0 scikit-image==0.14.1
```

## Citation

B. K. Iwana, R. Kuroki, and S. Uchida, "Explaining Convolutional Neural Networks using Softmax Gradient Layer-wise Relevance Propagation," in International Conference on Computer Vision Workshops, 2019.

```
@inproceedings{iwana2019explaining,
  title={Explaining Convolutional Neural Networks using Softmax Gradient Layer-wise Relevance Propagation},
  author={Iwana, Brian Kenji and Kuroki, Ryohei and Uchida, Seiichi},
  booktitle={International Conference on Computer Vision Workshops},
  year={2019}
}
```
