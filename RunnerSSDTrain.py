import os
import time
import tensorflow as tf
from nets import ssd_vgg_300
from datasets import pascalvoc_2007
import tensorflow.contrib.slim as slim
from preprocessing import ssd_vgg_preprocessing


class RunnerTrain(object):

    def __init__(self, dataset_split_name="train", dataset_dir="./data/train", num_class=21, batch_size=4,
                 img_shape=(300, 300), net_model=ssd_vgg_300, data_format='NHWC', dataset_name=pascalvoc_2007,
                 ckpt_path='./models/ssd_vgg_300', ckpt_name="ssd_300_vgg.ckpt",
                 learning_rate=0.01, end_learning_rate=0.0001,
                 weight_decay=0.00004, negative_ratio=3., loss_alpha=1., label_smoothing=0.0):

        # 数据相关参数
        self.num_class = num_class
        self.dataset_dir = dataset_dir
        self.data_format = data_format
        self.dataset_name = dataset_name
        self.dataset_split_name = dataset_split_name
        self.dataset = dataset_name.get_split(dataset_split_name, dataset_dir, None, None)

        # 训练相关参数
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.end_learning_rate = end_learning_rate
        self.weight_decay = weight_decay
        self.negative_ratio = negative_ratio
        self.loss_alpha = loss_alpha
        self.label_smoothing = label_smoothing
        self.decay_steps = int(self.dataset.num_samples / self.batch_size * 2.0)
        self.learning_rate_decay_factor = 0.94
        self.num_epochs_per_decay = 2.0
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.opt_epsilon = 1.0

        # 模型相关参数
        self.img_shape = img_shape
        self.ssd_params = net_model.SSDNet.default_params._replace(num_classes=self.num_class)
        self.ssd_params = self.ssd_params._replace(img_shape=self.img_shape)

        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)
        self.ckpt_path = ckpt_path
        self.ckpt_name = ckpt_name

        # 网络和default boxes
        self.ssd_net = net_model.SSDNet(self.ssd_params)
        self.ssd_anchors = self.ssd_net.anchors(self.img_shape)

        # 数据：预处理，encode，批次
        # g_scores是（当前默认框与真实框的交）占（真实框）的比例
        image, g_classes, g_localisations, g_scores = self._get_data_tensor(self.dataset, self.batch_size, self.data_format)

        # train_op, r_total_loss, r_predictions, r_localisations, r_logits, r_end_points,
        # learning_rate, image, g_classes, g_localisations, g_scores, global_step
        self.net_tensor = self._get_net_tensor(data_format, image, g_classes, g_localisations, g_scores)

        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
        # 默认保存：全局变量和可保存变量(ops.GraphKeys.GLOBAL_VARIABLES,ops.GraphKeys.SAVEABLE_OBJECTS)
        self.saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=1.0, write_version=2)
        pass

    def train_demo(self, num_batches=1000, print_1_freq=2, save_model_freq=200):
        def func_print(run_result, batch_index, run_time):
            if batch_index % print_1_freq == 0:
                self.print_info("{} time={} : loss={} learn_rate={} global_step={}".format(
                    batch_index, run_time, run_result[1], run_result[6], run_result[-1]))
            if batch_index % save_model_freq == 0:
                # 3. 保存的时候添加上正确的全局步长
                global_step = tf.get_collection(tf.GraphKeys.GLOBAL_STEP)[0]
                self.saver.save(self.sess, os.path.join(self.ckpt_path, self.ckpt_name), global_step=global_step)
                self.print_info("{} saved model in ={}".format(batch_index, self.ckpt_path))
            pass

        def func_final_print(run_result):
            self.print_info("OVER. loss={} lr={}".format(run_result[1], run_result[6]))
            pass
        # train_op, r_total_loss, r_predictions, r_localisations, r_logits, r_end_points,
        # learning_rate, image, g_classes, g_localisations, g_scores
        self.train(self.net_tensor, func_print=func_print, func_final_print=func_final_print, num_batches=num_batches)
        pass

    def train(self, run_list, func_print, func_final_print, num_batches):
        self.sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        # 恢复模型
        self.restore_if_y(log_dir=self.ckpt_path)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)

        results = []
        for batch_index in range(num_batches):
            start_time = time.time()
            results = self.sess.run(run_list)
            run_time = time.time() - start_time
            if func_print is not None:  # 打印
                func_print(results, batch_index, run_time)
            pass

        # 最终结果
        if func_final_print is not None:
            func_final_print(results)
        else:
            self.print_info("over")
            pass

        coord.request_stop()
        coord.join(threads)
        pass

    # 获取数据
    def _get_data_tensor(self, dataset, batch_size, data_format):
        # 数据
        provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset, common_queue_capacity=20 * batch_size, common_queue_min=10 * batch_size, shuffle=True)
        # 提取数据
        [image, labels, bboxes] = provider.get(['image', 'object/label', 'object/bbox'])

        image, labels, bboxes = ssd_vgg_preprocessing.preprocess_for_train(image, labels, bboxes,
                                                                           self.img_shape, data_format)

        # 编码label和boxes：Encode ground-truth labels and bboxes.
        classes, localisations, scores = self.ssd_net.bboxes_encode(labels, bboxes, self.ssd_anchors)

        # reshape_list：拉直
        batch_tensors = self._reshape_list([image, classes, localisations, scores])
        r = tf.train.batch(batch_tensors, batch_size=batch_size, capacity=5 * batch_size)

        # reshape_list：变成原来的形状
        return self._reshape_list(r, shape=[1] + [len(self.ssd_anchors)] * 3)

    # 获取网络输出
    def _get_net_tensor(self, data_format, image, g_classes, g_localisations, g_scores):
        with slim.arg_scope(self.ssd_net.arg_scope(weight_decay=self.weight_decay, data_format=data_format)):
            r_predictions, r_localisations, r_logits, r_end_points = self.ssd_net.net(image, is_training=True)
            # Add loss function.
            self.ssd_net.losses(r_logits, r_localisations, g_classes, g_localisations, g_scores,
                                negative_ratio=self.negative_ratio, alpha=self.loss_alpha,
                                label_smoothing=self.label_smoothing)
            total_loss = tf.get_collection(tf.GraphKeys.LOSSES)
            r_total_loss = tf.add_n(total_loss, name='total_loss')

        # 1. 全局步长(属于ops.GraphKeys.GLOBAL_VARIABLES和tf.GraphKeys.GLOBAL_STEP)
        # 该值可以保存(saver默认保存全局变量和可保存变量(ops.GraphKeys.GLOBAL_VARIABLES,ops.GraphKeys.SAVEABLE_OBJECTS))
        # 在恢复模型的时候，将tf.GraphKeys.GLOBAL_STEP添加到需要恢复的变量就会被读取。
        global_step = tf.train.get_or_create_global_step()
        learning_rate = tf.train.polynomial_decay(self.learning_rate, global_step, self.decay_steps,
                                                  self.end_learning_rate, power=1.0, cycle=False)
        # learning_rate = tf.train.exponential_decay(self.learning_rate, global_step, self.decay_steps,
        #                                            self.learning_rate_decay_factor, staircase=True,
        #                                            name='exponential_decay_learning_rate')

        # 使用RMSP损失会出现nan
        # optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.9, epsilon=1.0)
        # 使用Adam可正常训练
        optimizer = tf.train.AdamOptimizer(learning_rate, beta1=self.adam_beta1,
                                           beta2=self.adam_beta2, epsilon=self.opt_epsilon)

        train_op = optimizer.minimize(r_total_loss, global_step, var_list=tf.trainable_variables())

        return [train_op, r_total_loss, r_predictions, r_localisations, r_logits, r_end_points,
                learning_rate, image, g_classes, g_localisations, g_scores, global_step]
        pass

    # shape=None,则将l拉成一维list，否则将一维list按照shape转换。
    @staticmethod
    def _reshape_list(l, shape=None):
        """Reshape list of (list): 1D to 2D or the other way around.

        Args:
          l: List or List of list.
          shape: 1D or 2D shape.
        Return
          Reshaped list.
        """
        r = []
        if shape is None:
            # Flatten everything.
            for a in l:
                if isinstance(a, (list, tuple)):
                    r = r + list(a)
                else:
                    r.append(a)
        else:
            # Reshape to list of list.
            i = 0
            for s in shape:
                if s == 1:
                    r.append(l[i])
                else:
                    r.append(l[i:i + s])
                i += s
        return r

    def restore_if_y(self, log_dir):
        # 加载模型
        ckpt = tf.train.get_checkpoint_state(log_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # 恢复可训练变量
            var_list = tf.trainable_variables()
            # 2. 恢复全局步长变量（若设置全局步长变量为可训练，则不需要这一步，在获取可训练变量的时候就会获取到）
            global_step = tf.get_collection(tf.GraphKeys.GLOBAL_STEP)
            if global_step:
                var_list.extend(global_step)
            tf.train.Saver(var_list=var_list).restore(self.sess, ckpt.model_checkpoint_path)

            result = self.sess.run(global_step)
            print(result)

            self.print_info("Restored model parameters from {}".format(ckpt.model_checkpoint_path))
        else:
            self.print_info('No checkpoint file found.')
            pass
        pass

    @staticmethod
    def print_info(info):
        print("{} {}".format(time.strftime("%H:%M:%S", time.localtime()), info))
        pass

    pass


if __name__ == '__main__':
    runner = RunnerTrain()
    runner.train_demo(num_batches=100000, print_1_freq=100, save_model_freq=2000)