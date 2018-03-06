"""
Convert a Caffe model file to TensorFlow checkpoint format.

Assume that the network built is a equivalent (or a sub-) to the Caffe definition.
"""
import tensorflow as tf
import tensorflow.contrib.slim as slim

from nets import ssd_vgg_300, ssd_vgg_512
from changemodels import caffe_scope


class ConvertCaffeToTensorflow(object):

    def __init__(self, net_model, caffemodel_path, num_class=21, ckpt_path=None):
        self.net_model = net_model.SSDNet
        self.caffemodel_path = caffemodel_path
        self.num_class = num_class
        self.ckpt_path = self.caffemodel_path.replace('.caffemodel', '.ckpt') if ckpt_path is None else ckpt_path

        self.net_model.default_params._replace(num_classes=self.num_class)
        self.ssd_net = self.net_model(self.net_model.default_params)
        pass

    def convert(self):
        # Caffe scope...
        caffemodel = caffe_scope.CaffeScope()
        caffemodel.load(self.caffemodel_path)

        with tf.Graph().as_default():
            # Image placeholder and model.
            img_input = tf.placeholder(shape=(1, self.net_model.default_params.img_shape[0],
                                              self.net_model.default_params.img_shape[1], 3),  dtype=tf.float32)
            # Create model.
            with slim.arg_scope(self.ssd_net.arg_scope_caffe(caffemodel)):
                self.ssd_net.net(img_input, is_training=False)
                pass

            with tf.Session() as session:
                # Run the init operation.
                session.run(tf.global_variables_initializer())
                # Save model in checkpoint.
                tf.train.Saver().save(session, self.ckpt_path, write_meta_graph=False)
            pass

        pass

    pass


if __name__ == '__main__':
    ConvertCaffeToTensorflow(net_model=ssd_vgg_300,
                             caffemodel_path="../caffemodels/VGG_VOC0712_SSD_300x300/"
                                             "VGG_VOC0712_SSD_300x300_iter_120000.caffemodel",
                             ckpt_path="../checkpoints/VGG_VOC0712_SSD_300x300.ckpt").convert()

    # ConvertCaffeToTensorflow(net_model=ssd_vgg_512,
    #                          caffemodel_path="../caffemodels/VGG_VOC0712Plus_SSD_512x512_ft/"
    #                                          "VGG_VOC0712Plus_SSD_512x512_ft_iter_160000.caffemodel",
    #                          ckpt_path="../checkpoints/VGG_VOC0712Plus_SSD_512x512.ckpt").convert()

