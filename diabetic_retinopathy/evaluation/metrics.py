import tensorflow as tf
import keras

class ConfusionMatrix(keras.metrics.Metric):

    def __init(self, num_classes, name="confusion_matrix", **kwargs):
        super(ConfusionMatrix, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes

        self.confusion_matrix = self.add_weight(
            "confusion_matrix", shape = (num_classes, num_classes), initializer="zeros", dtype=tf.int32
        )
    def update_state(self,  y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=1)

        cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=self.num_classes, dtype=tf.int32)
        self.confusion_matrix.assign_add(cm)

    def result(self):
        return self.confusion_matrix

