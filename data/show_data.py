import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim

from datasets import pascalvoc_2007


class ShowImage(object):

    def __init__(self, dataset_dir='../data/test', split_name='test', batch_size=16):
        self.dataset_dir = dataset_dir
        self.split_name = split_name
        self.batch_size = batch_size

        self.dataset = pascalvoc_2007.get_split(self.split_name, self.dataset_dir)
        self.provider = slim.dataset_data_provider.DatasetDataProvider(
            self.dataset, shuffle=False,common_queue_capacity=2 * self.batch_size, common_queue_min=self.batch_size)

        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
        pass

    def show(self):
        # Start populating queues.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)

        print('Dataset:', self.dataset.data_sources, '|', self.dataset.num_samples)

        image, shape, bboxes, labels = self.provider.get(['image', 'shape', 'object/bbox', 'object/label'])
        image_crop = tf.slice(image, [0, 0, 0], [250, 250, 3])
        print('Original vs crop:', image.get_shape(), image_crop.get_shape())

        # Draw ground truth bounding boxes using TF routine.
        image_bboxes = tf.squeeze(tf.image.draw_bounding_boxes(tf.expand_dims(tf.to_float(image) / 255., 0),
                                                               tf.expand_dims(bboxes, 0)))

        for i in range(1):
            # Eval and display the image + bboxes.
            result_img, result_shape, result_bboxes, result_labels = self.sess.run([image_bboxes, shape, bboxes, labels])

            # cv2.imshow("image", result_img)
            # cv2.waitKey(0)

            plt.figure(figsize=(10, 10))
            plt.imshow(result_img)
            plt.show()

            print('Image shape:', result_img.shape, result_shape)
            print('Bounding boxes:', result_bboxes)
            print('Labels:', result_labels)

        # 请求线程终止
        coord.request_stop()
        coord.join(threads)
        pass

    pass


if __name__ == '__main__':
    ShowImage(dataset_dir='../data/train', split_name='train').show()

