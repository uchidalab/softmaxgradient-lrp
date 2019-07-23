import numpy as np
from keras import backend as K


class GradCAM(object):
    def __init__(self,
                 model,
                 target_id,
                 layer_name="block5_pool",
                 relu=False,
                 **kwargs):

        class_output = model.output[:, target_id]

        conv_output = model.get_layer(
            layer_name).output  
        grads = K.gradients(class_output, conv_output)[
            0]  
        self.gradient_function = K.function(
            [model.input],
            [conv_output, grads],
        ) 
        self.relu = relu

    def analyze(self, inputs):
        outputs, grads_vals = self.gradient_function([inputs])

        weights = np.mean(grads_vals, axis=(1, 2))
        cams = (outputs * weights[:, np.newaxis, np.newaxis, :]).sum(
            axis=3, keepdims=True)

        if self.relu:
            cams = np.maximum(cams, 0)
        return cams