import os
import inputs
import tensorflow as tf
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


class MaxEpoch(Exception):
    pass


model_config = collections.OrderedDict({'embed': embedding_size,
                                        'rh' : hidden_size,
                                        'l' : learning_rate,
                                        'gc' : grad_clip,
                                        'option' : option
                                        })

model_dir = ['{}-{}'.format(key, model_config[key]) for key in model_config.keys()]
model_dir = '_'.join(model_dir)

vocabulary_size = inputs.vocab_size + 1 # for padding
id_to_word = inputs.id_to_word

config = tf.ConfigProto()
config.gpu_options.allow_growth = True


inputs_pl = tf.placeholder(tf.int32, shape=[None, None])
inputs_seq_len_pl = tf.placeholder(tf.int32, shape=[None])


with tf.name_scope('train'):
    train_inputs, train_init_op = inputs.inputs('train', batch_size)
    with tf.variable_scope('Model'):
        trn_model = model.Sequential_VAE('train',
            inputs_pl, inputs_seq_len_pl, vocabulary_size, embedding_size,
            hidden_size, learning_rate, grad_clip)

with tf.name_scope('Test'):
    test_inputs, test_init_op = inputs.inputs('test', batch_size)
    with tf.variable_scope('Model', reuse=True):
        test_model = model.Sequential_VAE( 'test',
            inputs_pl, inputs_seq_len_pl, vocabulary_size, embedding_size,
                 hidden_size, learning_rate, grad_clip)

count = 0
epoch = 0
best_loss = 1e4
saver = tf.train.Saver()
if not os.path.exists('model/{}'.format(model_dir)):
    os.makedirs('model/{}'.format(model_dir))

with tf.Session(config=config) as sess:
    summary_writer = tf.summary.FileWriter('summary/{}'.format(model_dir), sess.graph,
                                           flush_secs=10)
    sess.run(tf.global_variables_initializer())
    while(True):

        sess.run(train_init_op)
        sess.run(tf.local_variables_initializer())
        start = time.time()
        while (True):
            try:
                batch, seq_len = sess.run(train_inputs)
                feed_dict = {inputs_pl: batch, inputs_seq_len_pl: seq_len}
                _, summary_val_trn = sess.run([trn_model.train_op,
                                              trn_model.update_op_list],
                          feed_dict=feed_dict)
                count += 1
                if count % 100 == 0:
                    s_id, g_norm, xent, logits, pred, sum_val \
                        = sess.run([trn_model.sample_id,
                                    trn_model.global_grad_norm,
                                    trn_model.xent_masked,
                                    trn_model.logits,
                                    trn_model.pred,
                                    trn_model.summary_list],feed_dict=feed_dict)
                    print('epoch ', epoch, 'count ', count, 'grad_norm',g_norm)
                    print('sample : ', [id_to_word[x] for x in s_id[0][:seq_len[0] + 5]])
                    print('pred : ', [id_to_word[x] for x in pred[0][:seq_len[0] + 5]])
                    print('answer : ', [id_to_word[x] for x in batch[0][:seq_len[0] + 5]])
            except tf.errors.OutOfRangeError:
                print("epoch took {:.3f} minutes".format((time.time() - start) / 60))
                sess.run(test_init_op)
                while(True):
                    try:
                        batch, seq_len = sess.run(test_inputs)
                        feed_dict = {inputs_pl: batch, inputs_seq_len_pl: seq_len}
                        summary_val_test = sess.run(test_model.update_op_list,
                                               feed_dict=feed_dict)
                    except tf.errors.OutOfRangeError:
                        print('done eval')
                        print('trn loss', summary_val_trn)
                        print('test loss ', summary_val_test)
                        saver.save(sess, 'model/{}/model.ckpt'.format(model_dir),
                                   global_step=epoch)
                        trn_summary, test_summary = sess.run([trn_model.summary,
                                                     test_model.summary])

                        summary_writer.add_summary(trn_summary, epoch)
                        summary_writer.add_summary(test_summary, epoch)
                        break
                break

        epoch += 1
        if epoch == max_epoch:
            raise MaxEpoch('Max Epoch Reached')

