import time
import tensorflow as tf
import tf_extend as tfe
from nets import ssd_vgg_300
from datasets import pascalvoc_2007
import tensorflow.contrib.slim as slim
from preprocessing import ssd_vgg_preprocessing


class RunnerEval(object):

    def __init__(self, batch_size=2, num_class=21, net_model=ssd_vgg_300, image_shape=(300, 300),
                 dataset_name=pascalvoc_2007, dataset_dir="./data/test", dataset_split_name="test",
                 eval_resize=4, data_format="NHWC", ckpt_path="./checkpoints/ssd_300_vgg.ckpt",
                 matching_threshold=0.5, select_threshold=0.01, select_top_k=400, keep_top_k=200, nms_threshold=0.45):

        # 参数
        with tf.name_scope("param"):
            # 数据相关参数
            self.num_class = num_class
            self.dataset_dir = dataset_dir
            self.data_format = data_format
            self.dataset_name = dataset_name
            self.dataset_split_name = dataset_split_name
            self.dataset = dataset_name.get_split(dataset_split_name, dataset_dir, None, None)

            # 模型相关参数
            self.net_model = net_model
            self.image_shape = image_shape
            self.ckpt_path = ckpt_path
            self.ckpt_path = tf.train.latest_checkpoint(ckpt_path) if tf.gfile.IsDirectory(ckpt_path) else ckpt_path
            self.net_params = net_model.SSDNet.default_params
            self.net_params._replace(num_classes=self.num_class)
            self.net_params._replace(img_shape=self.image_shape)

            # 验证相关参数
            self.batch_size = batch_size
            self.eval_resize = eval_resize
            self.max_batches = self.dataset.num_samples // self.batch_size

            # 选择边界框相关参数
            self.matching_threshold = matching_threshold
            self.select_threshold = select_threshold
            self.select_top_k = select_top_k
            self.keep_top_k = keep_top_k
            self.nms_threshold = nms_threshold
            pass

        # 网络和default boxes
        self.ssd_net = net_model.SSDNet(self.net_params)
        # anchors[0]和anchors[1]按顺序记录每个点的坐标。anchors[2]和anchors[3]记录了k个默认框的高度和宽度
        self.ssd_anchors = self.ssd_net.anchors(self.image_shape)

        # 数据：预处理，encode，批次
        # g_scores是（当前默认框与真实框的交）占（真实框）的比例
        image, g_labels, g_bboxes, g_diff, g_bbox_img, g_classes, g_localisations, g_scores = self._get_data_tensor(
            self.dataset, batch_size=self.batch_size, data_format=self.data_format, eval_resize=self.eval_resize)

        # 网络：net,loss
        r_predictions, r_localisations, r_logits, r_end_points, r_total_loss = self._get_net_tensor(
            self.ssd_net, self.data_format, image, g_classes, g_localisations, g_scores)

        # 解码，筛选
        # Performing post-processing on CPU: loop-intensive, usually more efficient.
        with tf.device('/device:CPU:0'):
            r_localisations = self.ssd_net.bboxes_decode(r_localisations, self.ssd_anchors)
            r_scores, r_bboxes = self.ssd_net.detected_bboxes(r_predictions, r_localisations, self.select_threshold,
                                                              self.nms_threshold, None, self.select_top_k, self.keep_top_k)
            # Metrics
            self.metrics = self._get_metrics_tensor(r_scores, r_bboxes, g_labels, g_bboxes,
                                                    g_diff, self.matching_threshold)
            pass

        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
        # g_labels, g_bboxes, g_classes, g_localisations, g_scores
        # names_to_values, names_to_updates, num_g_bboxes, tp_fp_metric, aps_voc07,
        # aps_voc12, mAP_voc_07, mAP_voc_12, r_total_loss, r_scores, r_bboxes
        self.run_list = [g_labels, g_bboxes, g_classes, g_localisations, g_scores]
        self.run_list = self.run_list + self.metrics + [r_total_loss, r_scores, r_bboxes]
        pass

    def eval_demo(self, num_batches=None, print_1_freq=2, print_2_freq=20):
        """
        eval demo
        
        :param num_batches: num_batches=None表示运行所有，num_batches=n表示运行n个批次
        :param print_1_freq: 
        :param print_2_freq: 
        :return: 
        """

        def func_print(run_result, batch_index, run_time):
            if batch_index % print_1_freq == 0:
                print("{} time={} : var_loss={} mAP_voc_07={} mAP_voc_12={}".format(
                    batch_index, run_time, run_result[13], run_result[11], run_result[12]))
            if batch_index % print_2_freq == 0:
                print("aps_voc07={}".format(run_result[9]))
                print("aps_voc12={}".format(run_result[10]))
            pass

        def func_final_print(run_result):
            print("aps_voc07={}".format(run_result[9]))
            print("aps_voc12={}".format(run_result[10]))
            pass

        # g_labels, g_bboxes, g_classes, g_localisations, g_scores
        # names_to_values, names_to_updates, num_g_bboxes, tp_fp_metric, aps_voc07,
        # aps_voc12, mAP_voc_07, mAP_voc_12,
        run_list = self.run_list
        self.run(run_list, func_print=func_print, func_final_print=func_final_print, num_batches=num_batches)
        pass

    def run(self, run_list, func_print, func_final_print=None, num_batches=None):
        self.sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

        print("Evaluating {}".format(self.ckpt_path))
        tf.train.Saver(var_list=slim.get_variables_to_restore()).restore(sess=self.sess, save_path=self.ckpt_path)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)

        results = []
        for batch_index in range(num_batches if num_batches else self.max_batches):
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
            print("over")
            pass

        coord.request_stop()
        coord.join(threads)
        pass

    # 获取度量
    @staticmethod
    def _get_metrics_tensor(r_scores, r_bboxes, g_labels, g_bboxes, g_diff, matching_threshold):
        # Compute TP and FP statistics.
        num_g_bboxes, tp, fp, r_scores = tfe.bboxes_matching_batch(
            r_scores.keys(), r_scores, r_bboxes, g_labels, g_bboxes, g_diff, matching_threshold)
        # 字典
        dict_metrics, aps_voc07, aps_voc12 = {}, {}, {}
        # FP and TP metrics.
        tp_fp_metric = tfe.streaming_tp_fp_arrays(num_g_bboxes, tp, fp, r_scores)
        for c in tp_fp_metric[0].keys():
            dict_metrics['tp_fp_%s' % c] = (tp_fp_metric[0][c], tp_fp_metric[1][c])
        # 计算AP
        for c in tp_fp_metric[0].keys():
            prec, rec = tfe.precision_recall(*tp_fp_metric[0][c])
            aps_voc07[c] = tfe.average_precision_voc07(prec, rec)
            aps_voc12[c] = tfe.average_precision_voc12(prec, rec)
        # 计算mAP
        mAP_voc_07 = tf.add_n(list(aps_voc07.values())) / len(aps_voc07)
        mAP_voc_12 = tf.add_n(list(aps_voc12.values())) / len(aps_voc12)
        # Split into values and updates ops.
        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map(dict_metrics)

        return_list = [names_to_values, names_to_updates, num_g_bboxes, tp_fp_metric,
                       aps_voc07, aps_voc12, mAP_voc_07, mAP_voc_12]
        return return_list

    # 获取网络输出
    @staticmethod
    def _get_net_tensor(ssd_net, data_format, image, g_classes, g_localisations, g_scores):
        with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
            r_predictions, r_localisations, r_logits, r_end_points = ssd_net.net(image, is_training=False)
            ssd_net.losses(r_logits, r_localisations, g_classes, g_localisations, g_scores)
            r_total_loss = tf.add_n(tf.get_collection(tf.GraphKeys.LOSSES), name='total_loss')
            return r_predictions, r_localisations, r_logits, r_end_points, r_total_loss
        pass

    # 获取数据
    def _get_data_tensor(self, dataset, batch_size, data_format, eval_resize):
        # 数据
        provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset, common_queue_capacity=2 * batch_size, common_queue_min=batch_size, shuffle=False)
        # 提取数据
        [image, labels, bboxes, diff] = provider.get(['image', 'object/label', 'object/bbox', 'object/difficult'])

        # 数据增强,不移除difficults。传labels是因为移除difficults时需要标签
        image, labels, bboxes, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
            image, labels, bboxes, self.ssd_net.params.img_shape, data_format, resize=eval_resize, difficults=None)

        # 编码label和boxes：Encode ground-truth labels and bboxes.
        classes, localisations, scores = self.ssd_net.bboxes_encode(labels, bboxes, self.ssd_anchors)

        # reshape_list：拉直
        batch_tensors = self._reshape_list([image, labels, bboxes, diff, bbox_img, classes, localisations, scores])
        r = tf.train.batch(batch_tensors, batch_size=batch_size, capacity=5 * batch_size, dynamic_pad=True)

        # reshape_list：变成原来的形状
        return self._reshape_list(r, shape=[1] * 5 + [len(self.ssd_anchors)] * 3)

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

    pass

if __name__ == '__main__':
    runner = RunnerEval(ckpt_path="models/ssd_vgg_300_fine/ssd_300_vgg.ckpt-3000", batch_size=16)
    runner.eval_demo()
