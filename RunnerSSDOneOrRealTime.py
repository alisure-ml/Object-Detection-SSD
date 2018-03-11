import time
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from nets import ssd_vgg_300, ssd_vgg_512, np_methods
from preprocessing import ssd_vgg_preprocessing


"""
没有框编码和框解码、没有优化
"""


class RunnerOneOrRealTime(object):

    def __init__(self, ckpt_filename, net_model, num_class=23, net_shape=(300, 300), data_format="NHWC"):
        self.ckpt_filename = ckpt_filename
        self.data_format = data_format
        self.net_shape = net_shape
        self.num_class = num_class

        self.ssd_net = net_model.SSDNet(net_model.SSDNet.default_params._replace(num_classes=num_class))
        self.img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
        self.image_4d, self.predictions, self.localisations, self.bbox_img, self.ssd_anchors = self.net(
            self.ssd_net, self.img_input, self.net_shape, self.data_format)

        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
        self.saver = tf.train.Saver()
        pass

    @staticmethod
    def net(ssd_net, img_input, net_shape, data_format):
        # 数据预处理
        image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
            img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
        # 升维
        image_4d = tf.expand_dims(image_pre, 0)

        with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
            predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False, reuse=False)

        # 得到默认的bounding boxes
        ssd_anchors = ssd_net.anchors(net_shape)

        return image_4d, predictions, localisations, bbox_img, ssd_anchors

    def run_net(self, img, select_threshold=0.5, nms_threshold=0.45, bboxes_sort_top_k=400):
        # Run SSD network.
        r_img, r_predictions, r_localisations, r_bbox_img = self.sess.run(
            [self.image_4d, self.predictions, self.localisations, self.bbox_img], feed_dict={self.img_input: img})

        # 将符合条件（非背景得分大于select_threshold）框的类别、得分和边界框筛选出
        r_classes, r_scores, r_bboxes = np_methods.ssd_bboxes_select(
            r_predictions, r_localisations, self.ssd_anchors, select_threshold=select_threshold,
            img_shape=self.net_shape, num_classes=self.num_class, decode=True)

        # 使bboxes的范围在bbox_ref内
        r_bboxes = np_methods.bboxes_clip(r_bbox_img, r_bboxes)
        # 根据得分排序，选择top_k
        r_classes, r_scores, r_bboxes = np_methods.bboxes_sort(r_classes, r_scores, r_bboxes, top_k=bboxes_sort_top_k)
        # 非极大值抑制(non maximum suppression)
        r_classes, r_scores, r_bboxes = np_methods.bboxes_nms(r_classes, r_scores, r_bboxes, nms_threshold)
        # Resize bboxes to original image shape.
        r_bboxes = np_methods.bboxes_resize(r_bbox_img, r_bboxes)

        return r_classes, r_scores, r_bboxes

    # 输入原始图像，返回最终结果
    def run_func(self, frame):
        r_classes, r_scores, r_bboxes = self.run_net(frame)
        return self.add_boxes_to_image(frame, r_classes, r_scores, r_bboxes)

    # 可视化: 原图，类别，得分，框
    @staticmethod
    def add_boxes_to_image(frame, classes, scores, bboxes,
                           colors_table=list([(255, 255, 255), (0, 0, 255), (0, 255, 0), (255, 0, 0),
                                              (255, 255, 0), (255, 0, 255), (0, 255, 255), (128, 0, 255),
                                              (128, 255, 0), (255, 128, 0), (0, 128, 255), (0, 255, 128),
                                              (255, 0, 128), (0, 128, 128), (128, 0, 128), (128, 128, 0),
                                              (128, 128, 255), (128, 255, 128), (255, 128, 128), (64, 128, 255),
                                              (0, 0, 0)])):
        """
        Visualize bounding boxes. Largely inspired by SSD-MXNET!
        """
        # 创建一个掩膜为了后面绘制
        mask = np.zeros_like(frame)
        height, width = frame.shape[0], frame.shape[1]

        for i in range(classes.shape[0]):
            cls_id = int(classes[i])
            if cls_id >= 0:
                y_min, x_min, y_max, x_max = (int(bboxes[i, 0] * height), int(bboxes[i, 1] * width),
                                              int(bboxes[i, 2] * height), int(bboxes[i, 3] * width))

                # 画矩形
                cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), color=colors_table[cls_id], thickness=2)
                # 写字
                cv2.putText(mask, '{:s} | {:.3f}'.format(str(cls_id), scores[i]), org=(x_min, y_min - 10),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=colors_table[0], thickness=1)
            pass

        return cv2.add(frame, mask)

    # 从图片中读入数据
    @staticmethod
    def read_image(run_func, image_name, result_name=None):
        img = cv2.imread(image_name)
        win_name = 'result'
        cv2.namedWindow(win_name)
        start_time = time.clock()
        result = run_func(img)
        cv2.imshow(win_name, result)
        if result_name is not None:
            cv2.imwrite(result_name, result)
        print("all time is {}".format(time.clock() - start_time))
        cv2.waitKey(0)
        cv2.destroyWindow(win_name)
        pass

    # 从视频或者摄像头读入数据
    @staticmethod
    def video_capture(run_func, prop_id, size):
        is_camera = True if isinstance(prop_id, int) and size is not None else False
        cap = cv2.VideoCapture(prop_id)

        if is_camera:
            cap.set(3, size[0])
            cap.set(4, size[1])

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                if is_camera:  # 摄像头需要flip，而视频不需要
                    frame = cv2.flip(frame, 1)

                start_time = time.clock()
                cv2.imshow("result", run_func(frame))
                print("one time is {}".format(time.clock() - start_time))

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break
            pass

        cap.release()
        cv2.destroyAllWindows()
        pass

    # 运行入口
    def run(self, image_name=None, result_name=None, prop_id=0, size=(960, 840)):
        # 初始化并恢复模型
        self.sess.run(tf.global_variables_initializer())
        self.saver.restore(self.sess, self.ckpt_filename)

        if image_name is not None:
            # 读图片
            self.read_image(run_func=self.run_func, image_name=image_name, result_name=result_name)
        else:
            # 捕获图片->run->画框
            self.video_capture(run_func=self.run_func, prop_id=prop_id, size=size)
        pass

    pass


def demo_300(ckpt_filename='checkpoints/VGG_VOC0712_SSD_300x300.ckpt'):
    runner = RunnerOneOrRealTime(ckpt_filename=ckpt_filename,
                                 net_model=ssd_vgg_300, num_class=21, net_shape=(300, 300))
    # one image
    runner.run(image_name="demo/dog.jpg", result_name="demo/dog_result2.png")
    # camera
    # runner.run(prop_id=0, size=(960, 840))
    # video
    # runner.run(prop_id="demo/video1.mp4")
    pass


def demo_512():
    runner = RunnerOneOrRealTime(ckpt_filename='checkpoints/VGG_VOC0712Plus_SSD_512x512.ckpt',
                                 net_model=ssd_vgg_512, num_class=21, net_shape=(512, 512))
    # one image
    runner.run(image_name="demo/dog.jpg", result_name="demo/dog_result.png")
    # camera
    # runner.run(prop_id=0, size=(960, 840))
    # video
    # runner.run(prop_id="demo/video1.mp4")
    pass

if __name__ == '__main__':
    demo_300(ckpt_filename="models/ssd_vgg_300/ssd_300_vgg.ckpt-600")
    pass
