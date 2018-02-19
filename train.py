import inputs
import tensorflow as tf
import numpy as np
import model
import time
import argparse
import collections

parser = argparse.ArgumentParser()
parser.add_argument('-batch_size', type=int)
parser.add_argument('-embedding_size', type=int)
parser.add_argument('-hidden_size', type=int)
parser.add_argument('-learning_rate', type=float)
parser.add_argument('-grad_clip', type=float)
parser.add_argument('-max_epoch', type=int, default=50)
parser.add_argument('-option', type=str)

args = parser.parse_args()
batch_size = args.batch_size
embedding_size = args.embedding_size
hidden_size = args.hidden_size
learning_rate = args.learning_rate
grad_clip = args.grad_clip
max_epoch = args.max_epoch
option = args.option

model_config = collections.OrderedDict({'embed': embedding_size,
                                        'rh' : hidden_size,
                                        'l' : learning_rate,
                                        'gc' : grad_clip,
                                        'option' : option
                                        })

model_dir = ['{}-{}'.format(key, model_config[key]) for key in model_config.keys()]
model_dir = '_'.join(model_dir)


vocabulary_size = inputs.vocab_size + 1
id_to_word = inputs.id_to_word

config = tf.ConfigProto()
config.gpu_options.allow_growth = True


inputs_pl = tf.placeholder(tf.int32, shape=[None, None])
inputs_seq_len_pl = tf.placeholder(tf.int32, shape=[None])


with tf.name_scope('train'):
    train_inputs, train_init_op = inputs.inputs('train', batch_size)
    with tf.variable_scope('Model'):
        trn_model = model.Sequential_VAE(
            inputs_pl, inputs_seq_len_pl, vocabulary_size, embedding_size,
            hidden_size, learning_rate, grad_clip)

        # trn_model = model.Sequential_VAE(
        #     inputs_pl, inputs_seq_len_pl, vocabulary_size, embedding_size,
        #          hidden_size, learning_rate, grad_clip)

# with tf.name_scope('Test'):
#     test_inputs, test_init_op = inputs.inputs('test', batch_size)
#     with tf.variable_scope('Model'):
#         test_model = model.Sequential_VAE(
#             inputs_pl, inputs_seq_len_pl, vocabulary_size, embedding_size,
#                  hidden_size, learning_rate, grad_clip)

count = 0
epoch = 0
saver = tf.train.Saver()

with tf.Session(config=config) as sess:
    summary_writer = tf.summary.FileWriter('summary/{}'.format(model_dir), sess.graph,
                                           flush_secs=10)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    sess.run(train_init_op)
    start = time.time()
    while (True):
        try:
            batch, seq_len = sess.run(train_inputs)
        except tf.errors.OutOfRangeError:
            print("epoch took {:.3f} minutes".format((time.time() - start) / 60))
            sess.run(tf.local_variables_initializer())
            start = time.time()
            sess.run(train_init_op)

            while(True):
                try:
                    train_loss = sess.run(trn_model.summary_list)
                except tf.errors.OutOfRangeError:
                    print(train_loss)

            saver.save(sess, 'model/{}/model.ckpt'.format(model_dir),
                       global_step=epoch)
            summary = sess.run(trn_model.summary)
            summary_writer.add_summary(summary, epoch)

            epoch += 1
            count = 0
        feed_dict = {inputs_pl: batch, inputs_seq_len_pl: seq_len}
        sess.run([trn_model.train_op], feed_dict=feed_dict)
        count += 1
        if count % 1 == 0:
            s_id, loss, g_norm, xent, logits = sess.run([trn_model.sample_id, trn_model.loss,
                                    trn_model.global_grad_norm, trn_model.xent_masked,
                                                 trn_model.logits],
                                  feed_dict=feed_dict)
            print('epoch ', epoch, 'count ', count, 'loss', loss, 'grad_norm', g_norm)
            print('predic : ', [id_to_word[x] for x in s_id[0][:seq_len[0] +5]])
            print('answer : ', [id_to_word[x] for x in batch[0][:seq_len[0] + 5]])





