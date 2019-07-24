# Softmax Gradient Layer-wise Relevance Propagation (SGLRP)

This is a Keras implementation of the the paper *Explaining Convolutional Neural Networks using Softmax Gradient Layer-wise Relevance Propagation* by Ryouhei Kuroki, Brian Kenji Iwana, and Seiichi Uchida.

## Abstract

Convolutional Neural Networks (CNN) have become state-of-the-art models for image classification. However, not everything is understood about the inner representations of CNNs. This paper tackles the interpretability and explainability of the predictions of CNNs for multi-class classification problems. Specifically, we propose a novel visualization method of pixel-wise input attribution called Softmax-Gradient Layer-wise Relevance Propagation (SGLRP). The proposed model is a class discriminate extension to Deep Taylor Decomposition (DTD) using the gradient of softmax to back propagate the relevance of the output probability to the input image. Through qualitative and quantitative analysis, we demonstrate that SGLRP can successfully localize and attribute the regions on input images which contribute to a target object's classification. Furthermore, we show that the proposed method excels at discriminating the target objects class from the other possible objects in the images. We confirm that SGLRP performs better than existing Layer-wise Relevance Propagation (LRP) methods and can help in the understanding of the decision process of CNNs. 

## Usage

[Example Notebook](example.ipynb)

## Requires

This code was developed in Python 3.5.2.

```
pip install keras==2.2.4 numpy==1.14.5 matplotlib==2.2.2 innvestigate==1.0.0 scikit-image==0.14.1
```