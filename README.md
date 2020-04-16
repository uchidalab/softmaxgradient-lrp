# Softmax Gradient Layer-wise Relevance Propagation (SGLRP)

This is a Keras implementation of the the paper Brian Kenji Iwana, Ryouhei Kuroki, and Seiichi Uchida, *Explaining Convolutional Neural Networks using Softmax Gradient Layer-wise Relevance Propagation,* ICCV Workshops, 2019. [arXiv](https://arxiv.org/abs/1908.04351)

## Abstract

Convolutional Neural Networks (CNN) have become state-of-the-art in the field of image classification. However, not everything is understood about their inner representations. This paper tackles the interpretability and explainability of the predictions of CNNs for multi-class classification problems. Specifically, we propose a novel visualization method of pixel-wise input attribution called Softmax-Gradient Layer-wise Relevance Propagation (SGLRP). The proposed model is a class discriminate extension to Deep Taylor Decomposition (DTD) using the gradient of softmax to back propagate the relevance of the output probability to the input image. Through qualitative and quantitative analysis, we demonstrate that SGLRP can successfully localize and attribute the regions on input images which contribute to a target object's classification. We show that the proposed method excels at discriminating the target objects class from the other possible objects in the images. We confirm that SGLRP performs better than existing Layer-wise Relevance Propagation (LRP) based methods and can help in the understanding of the decision process of CNNs. 

## News

- 2020/01/26: SGLRPSeqA, SGLRPSeqB - New implentation of SGLRP based on Sequential LRP. White paper incoming, for now if you use SGLRPSeqA or SGLRPSeqB, cite the paper below. Warning, experimental, sometimes it doesn't work as expected.
- 2019/12/20: GPU docker support
- 2019/11/02: ICCV Workshop on Explainable AI
- 2019/08/06: ArXiv paper posted
- 2019/07/23: Initial commit

## Details

SGLRP is a class contrastive extension of LRP. The general idea is that a relevance *penalty* is propagated through the network to create the relevance heatmaps. 

Specifically, we use the gradient of softmax as the initial relevance signal for LRP. Or,
![sglrpdef](https://latex.codecogs.com/gif.latex?R_%7Bn%7D%5E%7B%28L%29%7D%20%3D%20%5Cfrac%7B%5Cpartial%20%5Chat%7By%7D_t%7D%7B%5Cpartial%20z_n%7D%20%3D%20%5Cleft%5C%7B%5Cbegin%7Bmatrix%7D%20%26%20%5Chat%7By%7D_t%281-%5Chat%7By%7D_t%29%20%26%20n%3Dt%20%5C%5C%20%26%20-%20%5Chat%7By%7D_t%5Chat%7By%7D_n%20%26%20%5Cmathrm%7Botherwise%7D%2C%20%5Cend%7Bmatrix%7D%5Cright.)

where *t* is the target class and *n* is the other classes.

## Requires

This code was developed in Python 3.5.2. and requires Tensorflow 1.10.0

### Normal Install

```
pip install keras==2.1.5 numpy==1.14.5 matplotlib==2.2.2 innvestigate==1.0.8.3 scikit-image==0.15.0
```

### Docker

```
sudo docker build -t sglrp .
docker run --runtime nvidia -rm -it -p 127.0.0.1:8888:8888 -v `pwd`:/work -w /work sglrp jupyter notebook --allow-root
```

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
