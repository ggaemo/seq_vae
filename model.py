import tensorflow as tf

class Sequential_VAE():
    def __init__(self, inputs, inputs_seq_len, vocabulary_size, embedding_size,
                 hidden_size, learning_rate, grad_clip):

        self.inputs = inputs
        self.vocabulary_size = vocabulary_size
        self.inputs_seq_len = inputs_seq_len
        self.max_len = tf.shape(self.inputs)[1]

        embedding_matrix = tf.Variable(
            tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))

        self.word_embeddings = tf.nn.embedding_lookup(embedding_matrix, inputs)

        self.encoder_outputs, self.encoder_state = self.build_encoder(self.word_embeddings,
                                                            hidden_size,
                                                      inputs_seq_len)

        self.sample_id, self.logits = self.build_decoder(hidden_size, self.encoder_state)

        self.loss = self.compute_loss(self.logits)



        self.accuracy = self.compute_metrics(self.logits)

        self.summary_list, self.update_op_list = self.build_summary({'nll': self.loss,
                                                                     'acc' :
                                                                         self.accuracy})

        with tf.control_dependencies(self.update_op_list.values()):
            self.train_op = self.build_train_op(self.loss, learning_rate, grad_clip)

        self.summary = tf.summary.merge([tf.summary.scalar(key, val) for key, val in \
                                         self.summary_list.items()])

    def build_encoder(self, inputs, hidden_size, sequence_length):
        '''
        :param inputs: encoder input(word embeddings)
        :param hidden_size: dimension of rnn hidden states
        :param sequence_length: vector of the lengths of each encoder input sequence
        :return: outputs, state [output of rnn]
        '''
        # create a BasicRNNCell
        rnn_cell = tf.nn.rnn_cell.GRUCell(hidden_size)

        # 'outputs' is a tensor of shape [batch_size, max_time, cell_state_size]

        # defining initial state
        initial_state = rnn_cell.zero_state(tf.shape(inputs)[0], dtype=tf.float32)

        # 'state' is a tensor of shape [batch_size, cell_state_size]
        outputs, state = tf.nn.dynamic_rnn(rnn_cell, inputs,
                                           initial_state=initial_state,
                                           dtype=tf.float32,
                                           sequence_length=sequence_length
                                          )
        return outputs, state


    def build_decoder(self, hidden_size, decoder_input):
        '''
        :param hidden_size: dimension of rnn hidden states
        :param decoder_input: decoder input (encoder output state)
        :return: sample_id, logits []
        '''
        decoder_cell = tf.nn.rnn_cell.GRUCell(hidden_size)  # 그냥 seq vae
        helper = tf.contrib.seq2seq.TrainingHelper(self.word_embeddings, tf.ones_like(
            self.inputs_seq_len) * self.max_len)
        decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper,
                                                  initial_state=decoder_input)
        final_outputs, final_state, final_sequence_lengths = \
            tf.contrib.seq2seq.dynamic_decode(
            decoder)
        output_layer = tf.layers.Dense(self.vocabulary_size, use_bias=False)
        sample_id = final_outputs.sample_id
        logits = output_layer(final_outputs.rnn_output)
        return sample_id, logits


    def compute_loss(self, logits):
        xent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.inputs,
                                                              logits=logits)
        mask = tf.sequence_mask(self.inputs_seq_len, self.max_len, xent.dtype)
        self.mask = mask
        xent_masked = xent * mask
        train_loss = tf.reduce_mean(tf.reduce_sum(xent_masked, axis=1))
        self.xent = xent
        self.xent_masked = xent_masked
        return train_loss


    def compute_metrics(self, logits):
        mask = tf.sequence_mask(self.inputs_seq_len, self.max_len, tf.int32)
        acc = tf.metrics.accuracy(labels=self.inputs, predictions=tf.argmax(logits,
                                                                            axis=2),
                                  weights=mask,
                                  metrics_collections='metrics',
                                  updates_collections='update')
        return acc



    def build_train_op(self, loss, learning_rate, grad_clip):
        params = tf.trainable_variables()
        gradients = tf.gradients(loss, params)
        clipped_gradients, global_grad_norm = tf.clip_by_global_norm(gradients,
                                                                     grad_clip)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        update_step = optimizer.apply_gradients(zip(clipped_gradients, params))
        self.global_grad_norm = global_grad_norm
        return update_step


    def build_summary(self, variable_list):
        summary_list = dict()
        update_op_list = dict()
        for key, value in variable_list.items():
            mean, update_op = tf.contrib.metrics.streaming_mean(value,
                                                                metrics_collections='loss',
                                                                updates_collections='update')
            update_op_list[key] = update_op
            summary_list[key] = mean
        return summary_list, update_op_list
