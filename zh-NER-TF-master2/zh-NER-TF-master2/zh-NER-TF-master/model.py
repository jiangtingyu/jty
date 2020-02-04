import numpy as np
import os, time, sys
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode

import data
import utils
import eval


class BiLSTM_CRF(object):
    def __init__(self, args, embeddings, tag2label, vocab, paths, config):
        #参数
        #64
        self.batch_size = args.batch_size
        #40
        self.epoch_num = args.epoch
        #300
        self.hidden_dim = args.hidden_dim
        #wordid
        self.embeddings = embeddings
        #True use CRF at the top layer. if False, use Softmax
        self.CRF = args.CRF
        #True update embedding during training
        self.update_embedding = args.update_embedding
        #0.5 dropout keep_prob
        self.dropout_keep_prob = args.dropout
        #Adam Adam/Adadelta/Adagrad/RMSProp/Momentum/SGD
        self.optimizer = args.optimizer
        #0.001 learning rate
        self.lr = args.lr
        #5.0 gradient clipping 梯度裁剪
        self.clip_grad = args.clip
        #标签 标注的标签
        self.tag2label = tag2label
        #标签数量
        self.num_tags = len(tag2label)
        #路径
        self.vocab = vocab
        #shuffle training data before each epoch True
        self.shuffle = args.shuffle
        #模型路径
        self.model_path = paths['model_path']
        #总结路径
        self.summary_path = paths['summary_path']
        #日志路径
        self.logger = utils.get_logger(paths['log_path'])
        #结果路径
        self.result_path = paths['result_path']
        self.config = config

    def build_graph(self):
        self.add_placeholders()
        self.lookup_layer_op()
        self.biLSTM_layer_op()
        self.softmax_pred_op()
        self.loss_op()
        self.trainstep_op()
        self.init_op()

    #add_placeholders方法添加占位符
    def add_placeholders(self):
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name="word_ids")
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")
        # dropout lr浮点型
        self.dropout_pl = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        self.lr_pl = tf.placeholder(dtype=tf.float32, shape=[], name="lr")

    #lookup_layer_op方法目标在于利用预训练或随机初始化的embedding矩阵将句子中的每个字从one-hot向量映射为低维稠密的字向量
    def lookup_layer_op(self):
        #创建变量words
        with tf.variable_scope("words"):
            #创建变量_word_embedding 初始值为词向量
            _word_embeddings = tf.Variable(self.embeddings,#3905*300的矩阵，矩阵元素均在-0.25到0.25之间
                                           dtype=tf.float32,
                                           trainable=self.update_embedding,
                                           name="_word_embeddings")
            #选取一个张量里面索引对应的元素
            word_embeddings = tf.nn.embedding_lookup(params=_word_embeddings,
                                                     ids=self.word_ids,
                                                     name="word_embeddings")
        #tf.nn.dropout()是tensorflow里面为了防止或减轻过拟合而使用的函数，它一般用在全连接层
        self.word_embeddings =  tf.nn.dropout(word_embeddings, self.dropout_pl)

    #biLSTM_layer_op方法 可以有效地使用过去和未来的输入信息并自动提取功能。
    def biLSTM_layer_op(self):
        #理解bilstm的关键在于反向lstm并不是像看起来那样从右往左传递信息，而是先将原来的输入逆序排列输入到正向lstm中，
        # 再将得到的输出结果逆序排列，便得到了所谓的“反向lstm”的输出
        #创建变量bi-lstm
        with tf.variable_scope("bi-lstm"):
            #构造LSTM单元，有300个神经元
            #前向RNN
            cell_fw = LSTMCell(self.hidden_dim)
            #后向RNN
            cell_bw = LSTMCell(self.hidden_dim)
            #实现动态RNN
            #outputs为(output_fw, output_bw)，是一个包含前向cell输出tensor和后向cell输出tensor组成的二元组。
            (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=self.word_embeddings,
                sequence_length=self.sequence_lengths,
                dtype=tf.float32)
            # 则output_fw将是形状为[batch_size, max_time, cell_fw.output_size],
            # 的张量, 则output_bw将是形状为[batch_size, max_time, cell_bw.output_size]
            #拼接张量
            # [batch_size, max_time, 600]-1是按行的意思
            output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)

            print(output.shape)

            #tf.nn.dropout()是tensorflow里面为了防止或减轻过拟合而使用的函数，它一般用在全连接层
            output = tf.nn.dropout(output, self.dropout_pl)
        #创建变量proj
        with tf.variable_scope("proj"):
            #创建一个新的向量
            #name：新变量或现有变量的名称。
            #shape：新变量或现有变量的形状。
            #dtype：新变量或现有变量的类型（默认为DT_FLOAT）。
            #ininializer：如果创建了则用它来初始化变量。
            # 该函数返回一个用于初始化权重的初始化程序 “Xavier” 。
            # 这个初始化器是用来保持每一层的梯度大小都差不多相同
            #W是600*7
            W = tf.get_variable(name="W",
                                shape=[2 * self.hidden_dim, self.num_tags],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                dtype=tf.float32)

            b = tf.get_variable(name="b",
                                shape=[self.num_tags],
                                initializer=tf.zeros_initializer(),
                                dtype=tf.float32)
            # output的形状为[batch_size,steps,cell_num]批次大小，步长，神经元个数=600
            # reshape的目的是为了跟w做矩阵乘法
            s = tf.shape(output)

            output = tf.reshape(output, [-1, 2*self.hidden_dim]) #-1就是未知值，是批次大小

            pred = tf.matmul(output, W) + b  #[batch_size,self.num_tags]

            self.logits = tf.reshape(pred, [-1, s[1], self.num_tags])

    def loss_op(self):
        #在一个条件随机场里面计算标签序列的log-likelihood，函数的目的是使用crf来计算损失
        # crf_log_likelihood作为损失函数
        # inputs：unary potentials,就是每个标签的预测概率值
        # tag_indices，这个就是真实的标签序列了
        # sequence_lengths,一个样本真实的序列长度，为了对齐长度会做些padding，但是可以把真实的长度放到这个参数里
        # 输出：log_likelihood:标量;transition_params,转移概率，如果输入没输，它就自己算个给返回
        if self.CRF:
            log_likelihood, self.transition_params = crf_log_likelihood(inputs=self.logits,
                                                                   tag_indices=self.labels,
                                                                   sequence_lengths=self.sequence_lengths)
            #计算在每个张量的平均值
            self.loss = -tf.reduce_mean(log_likelihood)

        else:
            # 交叉熵做损失函数
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                    labels=self.labels)
            mask = tf.sequence_mask(self.sequence_lengths)
            losses = tf.boolean_mask(losses, mask)
            self.loss = tf.reduce_mean(losses)

        tf.summary.scalar("loss", self.loss)

    def softmax_pred_op(self):
        if not self.CRF:
            self.labels_softmax_ = tf.argmax(self.logits, axis=-1)
            self.labels_softmax_ = tf.cast(self.labels_softmax_, tf.int32)

    def trainstep_op(self):
        with tf.variable_scope("train_step"):
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            if self.optimizer == 'Adam':
                optim = tf.train.AdamOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Adadelta':
                optim = tf.train.AdadeltaOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Adagrad':
                optim = tf.train.AdagradOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'RMSProp':
                optim = tf.train.RMSPropOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Momentum':
                optim = tf.train.MomentumOptimizer(learning_rate=self.lr_pl, momentum=0.9)
            elif self.optimizer == 'SGD':
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)
            else:
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)

            # minimize()实际上包含了两个步骤，即compute_gradients和apply_gradients，前者用于计算梯度，后者用于使用计算得到的梯度来更新对应的variable
            grads_and_vars = optim.compute_gradients(self.loss)
            grads_and_vars_clip = [[tf.clip_by_value(g, -self.clip_grad, self.clip_grad), v] for g, v in grads_and_vars]
            self.train_op = optim.apply_gradients(grads_and_vars_clip, global_step=self.global_step)

    def init_op(self):
        self.init_op = tf.global_variables_initializer()

    def add_summary(self, sess):
        """

        :param sess:
        :return:
        """
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.summary_path, sess.graph)

    def train(self, train, dev):

        """train_data的形状为[(['我',在'北','京'],['O','O','B-LOC','I-LOC'])...第一句话
                                 (['我',在'天','安','门'],['O','O','B-LOC','I-LOC','I-LOC'])...第二句话
                                  ( 第三句话 )  ] 总共有50658句话"""
        """

        :param train:
        :param dev:
        :return:
        """


        saver = tf.train.Saver(tf.global_variables())

        with tf.Session(config=self.config) as sess:
            sess.run(self.init_op)
            self.add_summary(sess)

            for epoch in range(self.epoch_num):
                self.run_one_epoch(sess, train, dev, self.tag2label, epoch, saver)

    def test(self, test):
        saver = tf.train.Saver()
        with tf.Session(config=self.config) as sess:
            self.logger.info('=========== testing ===========')
            saver.restore(sess, self.model_path)
            label_list, seq_len_list = self.dev_one_epoch(sess, test)
            self.evaluate(label_list, seq_len_list, test)

    # 用模型测试一个句子
    def demo_one(self, sess, sent):
        """

        :param sess:
        :param sent: 
        :return:
        """

        #batch_yield就是把输入的句子每个字的id返回，以及每个标签转化为对应的tag2label的值
        label_list = []
        for seqs, labels in data.batch_yield(sent, self.batch_size, self.vocab, self.tag2label, shuffle=False):
            label_list_, _ = self.predict_one_batch(sess, seqs)
            label_list.extend(label_list_)
        label2tag = {}
        for tag, label in self.tag2label.items():
            label2tag[label] = tag if label != 0 else label
        tag = [label2tag[label] for label in label_list[0]]
        return tag

    def run_one_epoch(self, sess, train, dev, tag2label, epoch, saver):
        """

        :param sess:
        :param train:
        :param dev:
        :param tag2label:
        :param epoch:
        :param saver:
        :return:
        """

        # 计算出多少个batch，计算过程：(50658+64-1)//64=792
        num_batches = (len(train) + self.batch_size - 1) // self.batch_size
        # 记录开始训练的时间
        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        # 产生每一个batch
        batches = data.batch_yield(train, self.batch_size, self.vocab, self.tag2label, shuffle=self.shuffle)
        for step, (seqs, labels) in enumerate(batches):

            sys.stdout.write(' processing: {} batch / {} batches.'.format(step + 1, num_batches) + '\r')
            step_num = epoch * num_batches + step + 1
            feed_dict, _ = self.get_feed_dict(seqs, labels, self.lr, self.dropout_keep_prob)
            _, loss_train, summary, step_num_ = sess.run([self.train_op, self.loss, self.merged, self.global_step],
                                                         feed_dict=feed_dict)
            if step + 1 == 1 or (step + 1) % 300 == 0 or step + 1 == num_batches:
                self.logger.info(
                    '{} epoch {}, step {}, loss: {:.4}, global_step: {}'.format(start_time, epoch + 1, step + 1,
                                                                                loss_train, step_num))

            self.file_writer.add_summary(summary, step_num)

            if step + 1 == num_batches:
                saver.save(sess, self.model_path, global_step=step_num)

        self.logger.info('===========validation / test===========')
        label_list_dev, seq_len_list_dev = self.dev_one_epoch(sess, dev)
        self.evaluate(label_list_dev, seq_len_list_dev, dev, epoch)

    # 占位符赋值
    def get_feed_dict(self, seqs, labels=None, lr=None, dropout=None):
        """

        :param seqs:
        :param labels:
        :param lr:
        :param dropout:
        :return: feed_dict
        """
        # seq_len_list用来统计每个样本的真实长度
        # word_ids就是seq_list，padding后的样本序列
        word_ids, seq_len_list = data.pad_sequences(seqs, pad_mark=0)
        # labels经过padding后，喂给feed_dict
        feed_dict = {self.word_ids: word_ids,
                     self.sequence_lengths: seq_len_list}
        if labels is not None:
            labels_, _ = data.pad_sequences(labels, pad_mark=0)
            feed_dict[self.labels] = labels_
        if lr is not None:
            feed_dict[self.lr_pl] = lr
        if dropout is not None:
            feed_dict[self.dropout_pl] = dropout

        return feed_dict, seq_len_list

    def dev_one_epoch(self, sess, dev):
        """

        :param sess:
        :param dev:
        :return:
        """
        label_list, seq_len_list = [], []
        # 获取一个批次的句子中词的id以及标签
        for seqs, labels in data.batch_yield(dev, self.batch_size, self.vocab, self.tag2label, shuffle=False):
            label_list_, seq_len_list_ = self.predict_one_batch(sess, seqs)
            label_list.extend(label_list_)
            seq_len_list.extend(seq_len_list_)
        return label_list, seq_len_list

    def predict_one_batch(self, sess, seqs):
        """

        :param sess:
        :param seqs:
        :return: label_list
                 seq_len_list
        """
        feed_dict, seq_len_list = self.get_feed_dict(seqs, dropout=1.0)

        if self.CRF:
            # transition_params代表转移概率，由crf_log_likelihood方法计算出
            logits, transition_params = sess.run([self.logits, self.transition_params],
                                                 feed_dict=feed_dict)
            label_list = []
            for logit, seq_len in zip(logits, seq_len_list):
                viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_params)
                label_list.append(viterbi_seq)
            return label_list, seq_len_list

        else:
            label_list = sess.run(self.labels_softmax_, feed_dict=feed_dict)
            return label_list, seq_len_list

    def evaluate(self, label_list, seq_len_list, data, epoch=None):
        """

        :param label_list:
        :param seq_len_list:
        :param data:
        :param epoch:
        :return:
        """
        label2tag = {}
        for tag, label in self.tag2label.items():
            label2tag[label] = tag if label != 0 else label

        model_predict = []
        for label_, (sent, tag) in zip(label_list, data):
            tag_ = [label2tag[label__] for label__ in label_]
            sent_res = []
            if  len(label_) != len(sent):
                print(sent)
                print(len(label_))
                print(tag)
            for i in range(len(sent)):
                sent_res.append([sent[i], tag[i], tag_[i]])
            model_predict.append(sent_res)
        epoch_num = str(epoch+1) if epoch != None else 'test'
        label_path = os.path.join(self.result_path, 'label_' + epoch_num)
        metric_path = os.path.join(self.result_path, 'result_metric_' + epoch_num)
        for _ in eval.conlleval(model_predict, label_path, metric_path):
            self.logger.info(_)

