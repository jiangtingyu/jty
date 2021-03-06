import tensorflow as tf
import numpy as np
import os, argparse, time, random
import model
import utils
import data

## Session configuration
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 编号为0的GPU对程序可见
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # default: 0 显示错误信息，屏蔽通知和警告
# 动态申请显存
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.2  # need ~700MB GPU memory 占用20%显存


## hyperparameters
parser = argparse.ArgumentParser(description='BiLSTM-CRF for Chinese NER task')
parser.add_argument('--train_data', type=str, default='data_path', help='train data source')
parser.add_argument('--test_data', type=str, default='data_path', help='test data source')
parser.add_argument('--batch_size', type=int, default=64, help='#sample of each minibatch')
parser.add_argument('--epoch', type=int, default=40, help='#epoch of training')
parser.add_argument('--hidden_dim', type=int, default=300, help='#dim of hidden state')
parser.add_argument('--optimizer', type=str, default='Adam', help='Adam/Adadelta/Adagrad/RMSProp/Momentum/SGD')
parser.add_argument('--CRF', type=utils.str2bool, default=True, help='use CRF at the top layer. if False, use Softmax')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout keep_prob')
parser.add_argument('--update_embedding', type=utils.str2bool, default=True, help='update embedding during training')
parser.add_argument('--pretrain_embedding', type=str, default='random', help='use pretrained char embedding or init it randomly')
parser.add_argument('--embedding_dim', type=int, default=300, help='random init char embedding_dim')
parser.add_argument('--shuffle', type=utils.str2bool, default=True, help='shuffle training data before each epoch')
#parser.add_argument('--mode', type=str, default='demo', help='train/test/demo')
parser.add_argument('--mode', type=str, default='train', help='train/test/demo')
parser.add_argument('--demo_model', type=str, default='1521112368', help='model for test and demo')
args = parser.parse_args()


## get char embeddings  获取预训练好的词向量
#'''word2id的形状为{'当': 1, '希': 2, '望': 3, '工': 4, '程': 5,。。'<UNK>': 3904, '<PAD>': 0}
# train_data总共3903个去重后的字'''
word2id = data.read_dictionary(os.path.join('.', args.train_data, 'word2id.pkl'))

#通过调用random_embedding函数返回一个len(vocab)*embedding_dim=3905*300的矩阵(矩阵元素均在-0.25到0.25之间)作为初始值
if args.pretrain_embedding == 'random':
    embeddings = data.random_embedding(word2id, args.embedding_dim)
else:
    embedding_path = 'pretrain_embedding.npy'
    embeddings = np.array(np.load(embedding_path), dtype='float32')


## read corpus and get training data  获取训练数据
if args.mode != 'demo':
    train_path = os.path.join('.', args.train_data, 'train_data')
    test_path = os.path.join('.', args.test_data, 'test_data')
    # 通过read_corpus函数读取出train_data
    """ train_data的形状为[(['我',在'北','京'],['O','O','B-LOC','I-LOC'])...第一句话
                         (['我',在'天','安','门'],['O','O','B-LOC','I-LOC','I-LOC'])...第二句话  
                          ( 第三句话 )  ] 总共有50658句话"""

    train_data = data.read_corpus(train_path)
    test_data = data.read_corpus(test_path); test_size = len(test_data)


## paths setting 路径设置
paths = {}

timestamp = str(int(time.time())) if args.mode == 'train' else args.demo_model

#输出地址,默认是./data_path_save/时间戳
output_path = os.path.join('.', args.train_data+"_save", timestamp)
if not os.path.exists(output_path): os.makedirs(output_path)

summary_path = os.path.join(output_path, "summaries")
paths['summary_path'] = summary_path
if not os.path.exists(summary_path): os.makedirs(summary_path)
model_path = os.path.join(output_path, "checkpoint/")
if not os.path.exists(model_path): os.makedirs(model_path)
ckpt_prefix = os.path.join(model_path, "model")
paths['model_path'] = ckpt_prefix
result_path = os.path.join(output_path, "results")
paths['result_path'] = result_path
if not os.path.exists(result_path): os.makedirs(result_path)
log_path = os.path.join(result_path, "log.txt")
paths['log_path'] = log_path
utils.get_logger(log_path).info(str(args))


## training model
if args.mode == 'train':
    # 引入第二步建立的模型
    model = model.BiLSTM_CRF(args, embeddings, data.tag2label, word2id, paths, config=config)
    # 创建节点，无返回值
    model.build_graph()

    ## hyperparameters-tuning, split train/dev
    # dev_data = train_data[:5000]; dev_size = len(dev_data)
    # train_data = train_data[5000:]; train_size = len(train_data)
    # print("train data: {0}\ndev data: {1}".format(train_size, dev_size))
    # model.train(train=train_data, dev=dev_data)

    ## train model on the whole training data
    print("train data: {}".format(len(train_data)))
    # 训练
    model.train(train=train_data, dev=test_data)  # use test_data as the dev_data to see overfitting phenomena

## testing model
elif args.mode == 'test':
    ckpt_file = tf.train.latest_checkpoint(model_path)
    print(ckpt_file)
    paths['model_path'] = ckpt_file
    model = model.BiLSTM_CRF(args, embeddings, data.tag2label, word2id, paths, config=config)
    model.build_graph()
    print("test data: {}".format(test_size))
    model.test(test_data)

## demo
elif args.mode == 'demo':
    ckpt_file = tf.train.latest_checkpoint(model_path)
    print(ckpt_file)
    paths['model_path'] = ckpt_file
    model = model.BiLSTM_CRF(args, embeddings, data.tag2label, word2id, paths, config=config)
    model.build_graph()
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        print('============= demo =============')
        saver.restore(sess, ckpt_file)
        while(1):
            print('Please input your sentence:')
            demo_sent = input()
            if demo_sent == '' or demo_sent.isspace():
                print('See you next time!')
                break
            else:
                demo_sent = list(demo_sent.strip())
                demo_data = [(demo_sent, ['O'] * len(demo_sent))]
                tag = model.demo_one(sess, demo_data)
                PER, LOC, ORG = utils.get_entity(tag, demo_sent)
                print('PER: {}\nLOC: {}\nORG: {}'.format(PER, LOC, ORG))
