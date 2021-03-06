"""
MIT License

Copyright (c) 2020 Hyeonki Hong <hhk7734@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from datetime import datetime
from os import path
import time
from typing import Union

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend, layers, models, optimizers

from . import dataset, train, weights
from ..model import yolov4
from ..utility import media, predict


class YOLOv4:
    def __init__(self, tiny: bool = False, tpu: bool = False):
        """
        Default configuration
        """
        self.tiny = tiny
        self.tpu = tpu

        if tiny:
            self.anchors = [
                [[23, 27], [37, 58], [81, 82]],
                [[81, 82], [135, 169], [344, 319]],
            ]
        else:
            self.anchors = [
                [[12, 16], [19, 36], [40, 28]],
                [[36, 75], [76, 55], [72, 146]],
                [[142, 110], [192, 243], [459, 401]],
            ]
        self.batch_size = 32
        self.subdivision = 16
        self._classes = None
        self._has_weights = False
        self.input_size = 608
        self.model = None
        if tiny:
            self.strides = [16, 32]
        else:
            self.strides = [8, 16, 32]
        if tiny:
            self.xyscales = [1.05, 1.05]
        else:
            self.xyscales = [1.2, 1.1, 1.05]

    @property
    def anchors(self):
        """
        Usage:
            yolo.anchors = [12, 16, 19, 36, 40, 28, 36, 75,
                            76, 55, 72, 146, 142, 110, 192, 243, 459, 401]
            yolo.anchors = np.array([12, 16, 19, 36, 40, 28, 36, 75,
                            76, 55, 72, 146, 142, 110, 192, 243, 459, 401])
            print(yolo.anchors)
        """
        return self._anchors

    @anchors.setter
    def anchors(self, anchors: Union[list, tuple, np.ndarray]):
        if isinstance(anchors, (list, tuple)):
            self._anchors = np.array(anchors)
        elif isinstance(anchors, np.ndarray):
            self._anchors = anchors

        if self.tiny:
            self._anchors = self._anchors.astype(np.float32).reshape(2, 3, 2)
        else:
            self._anchors = self._anchors.astype(np.float32).reshape(3, 3, 2)

    @property
    def classes(self):
        """
        Usage:
            yolo.classes = {0: 'person', 1: 'bicycle', 2: 'car', ...}
            yolo.classes = "path/classes"
            print(len(yolo.classes))
        """
        return self._classes

    @classes.setter
    def classes(self, data: Union[str, dict]):
        if isinstance(data, str):
            self._classes = media.read_classes_names(data)
        elif isinstance(data, dict):
            self._classes = data
        else:
            raise TypeError("YOLOv4: Set classes path or dictionary")

    @property
    def input_size(self):
        """
        Usage:
            yolo.input_size = 608
            print(yolo.input_size)
        """
        return self._input_size

    @input_size.setter
    def input_size(self, size: int):
        if size % 32 == 0:
            self._input_size = size
        else:
            raise ValueError("YOLOv4: Set input_size to multiples of 32")

    @property
    def strides(self):
        """
        Usage:
            yolo.strides = [8, 16, 32]
            yolo.strides = np.array([8, 16, 32])
            print(yolo.strides)
        """
        return self._strides

    @strides.setter
    def strides(self, strides: Union[list, tuple, np.ndarray]):
        if isinstance(strides, (list, tuple)):
            self._strides = np.array(strides)
        elif isinstance(strides, np.ndarray):
            self._strides = strides

    @property
    def xyscales(self):
        """
        Usage:
            yolo.xyscales = [1.2, 1.1, 1.05]
            yolo.xyscales = np.array([1.2, 1.1, 1.05])
            print(yolo.xyscales)
        """
        return self._xyscales

    @xyscales.setter
    def xyscales(self, xyscales: Union[list, tuple, np.ndarray]):
        if isinstance(xyscales, (list, tuple)):
            self._xyscales = np.array(xyscales)
        elif isinstance(xyscales, np.ndarray):
            self._xyscales = xyscales

    def make_model(
        self, activation0: str = "mish", activation1: str = "leaky",
    ):
        # pylint: disable=missing-function-docstring
        self._has_weights = False
        backend.clear_session()

        inputs = layers.Input([self.input_size, self.input_size, 3])
        if self.tiny:
            self.model = yolov4.YOLOv4Tiny(
                anchors=self.anchors,
                num_classes=len(self.classes),
                xyscales=self.xyscales,
                activation=activation1,
                tpu=self.tpu,
            )
        else:
            self.model = yolov4.YOLOv4(
                anchors=self.anchors,
                num_classes=len(self.classes),
                xyscales=self.xyscales,
                activation0=activation0,
                activation1=activation1,
            )
        self.model(inputs)

    def load_weights(self, weights_path: str, weights_type: str = "tf"):
        """
        Usage:
            yolo.load_weights("yolov4.weights", weights_type="yolo")
            yolo.load_weights("checkpoints")
        """
        if weights_type == "yolo":
            weights.load_weights(self.model, weights_path, tiny=self.tiny)
        elif weights_type == "tf":
            self.model.load_weights(weights_path)

        self._has_weights = True

    def save_as_tflite(self, tflite_path, quantization=None, data_set=None):
        """
        Save model and weights as tflite

        Usage:
            yolo.save_as_tflite("yolov4.tflite")
            yolo.save_as_tflite("yolov4-float16.tflite", "float16")
            yolo.save_as_tflite("yolov4-int.tflite", "int", data_set)
        """
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)

        def representative_dataset_gen():
            for _ in range(20):
                # pylint: disable=stop-iteration-return
                # TODO: # of iteration
                images, _ = next(data_set)
                yield [tf.cast(images[0:1, ...], tf.float32)]

        if quantization:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

        if quantization == "float16":
            converter.target_spec.supported_types = [tf.float16]
        elif quantization == "int":
            converter.representative_dataset = representative_dataset_gen
        elif quantization == "full_int8":
            converter.representative_dataset = representative_dataset_gen
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8
            ]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
        elif quantization:
            raise ValueError(
                "YOLOv4: {} is not a valid option".format(quantization)
            )

        tflite_model = converter.convert()
        with tf.io.gfile.GFile(tflite_path, "wb") as fd:
            fd.write(tflite_model)

    def resize_image(self, image, ground_truth=None):
        """
        @param image:        Dim(height, width, channels)
        @param ground_truth: [[center_x, center_y, w, h, class_id], ...]

        @return resized_image or (resized_image, resized_ground_truth)

        Usage:
            image = yolo.resize_image(image)
            image, ground_truth = yolo.resize_image(image, ground_truth)
        """
        return media.resize_image(
            image, target_size=self.input_size, ground_truth=ground_truth
        )

    def candidates_to_pred_bboxes(self, candidates):
        """
        @param candidates: Dim(-1, (x, y, w, h, conf, prob_0, prob_1, ...))

        @return Dim(-1, (x, y, w, h, class_id, probability))
        """
        return predict.candidates_to_pred_bboxes(candidates, self.input_size)

    def fit_pred_bboxes_to_original(self, pred_bboxes, original_shape):
        """
        @param pred_bboxes:    Dim(-1, (x, y, w, h, class_id, probability))
        @param original_shape: (height, width, channels)
        """
        # pylint: disable=no-self-use
        return predict.fit_pred_bboxes_to_original(pred_bboxes, original_shape)

    def draw_bboxes(self, image, bboxes):
        """
        @parma image:  Dim(height, width, channel)
        @param bboxes: (candidates, 4) or (candidates, 5)
                [[center_x, center_y, w, h, class_id], ...]
                [[center_x, center_y, w, h, class_id, propability], ...]

        @return drawn_image

        Usage:
            image = yolo.draw_bboxes(image, bboxes)
        """
        return media.draw_bboxes(image, bboxes, self.classes)

    #############
    # Inference #
    #############

    def predict(self, frame: np.ndarray):
        """
        Predict one frame

        @param frame: Dim(height, width, channels)

        @return pred_bboxes == Dim(-1, (x, y, w, h, class_id, probability))
        """
        # image_data == Dim(1, input_szie, input_size, channels)
        image_data = self.resize_image(frame)
        image_data = image_data / 255
        image_data = image_data[np.newaxis, ...].astype(np.float32)

        # s_pred, m_pred, l_pred
        # x_pred == Dim(1, output_size, output_size, anchors, (bbox))
        candidates = self.model.predict(image_data)
        _candidates = []
        for candidate in candidates:
            grid_size = candidate.shape[1]
            _candidates.append(
                tf.reshape(candidate, shape=(1, grid_size * grid_size * 3, -1))
            )
        candidates = np.concatenate(_candidates, axis=1)

        # Select 0
        pred_bboxes = self.candidates_to_pred_bboxes(candidates[0])
        pred_bboxes = self.fit_pred_bboxes_to_original(pred_bboxes, frame.shape)
        return pred_bboxes

    def inference(self, media_path, is_image=True, cv_waitKey_delay=10):
        if not path.exists(media_path):
            raise FileNotFoundError("{} does not exist".format(media_path))
        if is_image:
            frame = cv2.imread(media_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            start_time = time.time()
            bboxes = self.predict(frame)
            exec_time = time.time() - start_time
            print("time: {:.2f} ms".format(exec_time * 1000))

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            image = self.draw_bboxes(frame, bboxes)
            cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("result", image)
        else:
            vid = cv2.VideoCapture(media_path)
            while True:
                return_value, frame = vid.read()
                if return_value:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    break

                start_time = time.time()
                bboxes = self.predict(frame)
                curr_time = time.time()
                exec_time = curr_time - start_time
                info = "time: %.2f ms" % (1000 * exec_time)
                print(info)

                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                image = self.draw_bboxes(frame, bboxes)
                cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
                cv2.imshow("result", image)
                if cv2.waitKey(cv_waitKey_delay) & 0xFF == ord("q"):
                    break

        print("YOLOv4: Inference is finished")
        while cv2.waitKey(10) & 0xFF != ord("q"):
            pass
        cv2.destroyWindow("result")

    ############
    # Training #
    ############

    def load_dataset(
        self, dataset_path, dataset_type="converted_coco", training=True
    ):
        return dataset.Dataset(
            anchors=self.anchors,
            batch_size=self.batch_size // self.subdivision,
            dataset_path=dataset_path,
            dataset_type=dataset_type,
            data_augmentation=training,
            input_size=self.input_size,
            num_classes=len(self.classes),
            strides=self.strides,
            xyscales=self.xyscales,
        )

    def compile(
        self,
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss_iou_type: str = "ciou",
    ):
        self.model.compile(
            optimizer=optimizer,
            loss=train.YOLOv4Loss(
                batch_size=self.batch_size // self.subdivision,
                iou_type=loss_iou_type,
            ),
        )

    def fit(self, data_set, epochs, verbose=1, callbacks=None):
        # validation_split=0.,
        # validation_data=None,
        # shuffle=True,
        # class_weight=None,
        # sample_weight=None,
        # initial_epoch=0,
        # steps_per_epoch=None,
        # validation_steps=None,
        # validation_batch_size=None,
        # validation_freq=1,
        # max_queue_size=10,
        # workers=1,
        # use_multiprocessing=False
        self.model.fit(
            data_set,
            epochs=epochs,
            verbose=verbose,
            callbacks=callbacks,
            batch_size=self.batch_size // self.subdivision,
            steps_per_epoch=self.subdivision,
        )
