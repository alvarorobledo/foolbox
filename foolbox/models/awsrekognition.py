from __future__ import absolute_import

import numpy as np
import keras
import logging
import cv2
from concurrent.futures import ThreadPoolExecutor
from .base import Model
from .keras import KerasModel


class AWSRekognitionModel(Model):
    """Creates a :class:`Model` instance for the use of the AWS API.

    """
    def __init__(
            self,
            bounds,
            num_thread=4,
            channel_axis=3,
            preprocessing=(0, 1),
            predicts='probabilities'):

        super(AWSRekognitionModel, self).__init__(bounds=bounds,
                                            channel_axis=channel_axis,
                                            preprocessing=preprocessing)

        self._num_classes = 1000
        self._num_api_calls = 0
        self.localmodel = self.setup_local_model()
        self._pool = ThreadPoolExecutor(max_workers=num_thread)

    def batch_predictions(self, images):
        predictions = np.empty(shape=(len(images), self.num_classes()), dtype=np.float32)
        futures = []
        for image in images:
            futures.append(self._pool.submit(self.predictions, image))
        for i, future in enumerate(futures):
            predictions[i,] = future.result()
        return predictions

    def predictions(self, image):
        """
        Use AWS Rekognition API to annotate an image.
        :param image: a np.ndarray representing an RGB image.
        :return: a numpy array with shape (1,) to comply with the assumptions that a typical prediction (logits or probabilities) are 1D.
        """

        encoded_image = self.encode_image_AWS(image, extension='.png')
        decoded_image = self.decode_image_AWS(encoded_image)
        if not (np.array_equal(image, decoded_image)): #check if both are equal
            print('images not equal!')
            diff = image-encoded_image
            print(diff)
        #replace all of this when ready to get predictions from API
        arr = self.localmodel.predictions(image)
        # pred_labels = (-arr).argsort()[:5]
        # print(arr[pred_labels[0]],arr[pred_labels[1]])
        # print(pred_labels)
        pred = np.zeros((1), dtype=object)
        pred[0] = arr
        #self._num_api_calls += 1
        return arr
    
    def setup_local_model(self):
        #sets up local ResNet50 model, to use for local testing before wasting any AWS API calls
        keras.backend.set_learning_phase(0)
        kmodel = keras.applications.resnet50.ResNet50(weights='imagenet')
        preprocessing = (np.array([104, 116, 123]), 1)
        model = KerasModel(kmodel, bounds=(0, 255), preprocessing=preprocessing, predicts='logits')
        return model

    def num_classes(self):
        return self._num_classes

    def encode_image_AWS(self, input_image, extension='.png'):
        #encodes an image into the given extension (png by default)
        success, encoded_image = cv2.imencode(extension, input_image)
        if not success:
            print('not success')
            return False
        return encoded_image #reminder: need to send encoded_image.tobytes() to AWS

    def decode_image_AWS(self, encoded_image):
        #decodes a given image into a numpy array
        nparr = np.fromstring(encoded_image.tostring(), np.uint8)
        decoded_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return decoded_image