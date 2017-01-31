from nest.model import Model
from nest.utils import hyperdefault
import seq2seq_model as s2smodel
import tensorflow as tf
from seq_data import SeqData
import data_utils
import numpy as np
import time
import sys
import math

class SeqModel(Model):
    def __init__(self, sess, hyperparameters = {}, save_dir='./run/'):
        super().__init__(sess, hyperparameters, save_dir)
        self.model_name = "Seq2Seq"
        self.data_source = SeqData(save_dir)
        self.setup_hyperparameters()
        print("constructing model")
        self.construct()
        print("finished constructing")
        self.steps_per_checkpoint = 200

    def setup_hyperparameters(self):
        self.batch_size = hyperdefault("batch_size", 64, self.hyperparameters)
        self.layer_size = hyperdefault("layer_size", 256, self.hyperparameters)
        self.num_layers = hyperdefault("num_layer", 3, self.hyperparameters)
        self.learning_rate = hyperdefault("learning_rate", 0.5, self.hyperparameters)      
        self.learning_rate_decay_factor =  hyperdefault("learning_rate_decay", 0.99, self.hyperparameters)
        self.max_gradient_norm = hyperdefault("max_gradient_norm", 5.0, self.hyperparameters)
        self.num_batches = hyperdefault("num_batches", 1000, self.hyperparameters)
        self.steps_per_checkpoint = hyperdefault("steps_per_checkpoint", 200, self.hyperparameters)
    def construct(self):
        self.input_vocab_size = len(self.data_source.vocabA)
        self.output_vocab_size = len(self.data_source.vocabB)
        self.model = s2smodel.Seq2SeqModel( \
            self.input_vocab_size,
            self.output_vocab_size,
            self.data_source.buckets,
            self.layer_size,
            self.num_layers,
            self.max_gradient_norm,
            self.batch_size,
            self.learning_rate,
            self.learning_rate_decay_factor,
            forward_only = False,
            dtype=tf.float32)
        print("initializing variables")
        self.sess.run(tf.global_variables_initializer())

    def train_batch(self, X, y):
        super().train_batch(X, y)
        # Choose a bucket according to data distribution. We pick a random number
        # in [0, 1] and use the corresponding interval in train_buckets_scale.
        random_number_01 = np.random.random_sample()
        bucket_id = min([i for i in range(len(self.train_buckets_scale))
                         if self.train_buckets_scale[i] > random_number_01])

        # Get a batch and make a step.
        start_time = time.time()
        encoder_inputs, decoder_inputs, target_weights = self.model.get_batch(
            X, bucket_id)
        _, step_loss, _ = self.model.step(self.sess, encoder_inputs, decoder_inputs,
                                          target_weights, bucket_id, False)
        step_time = (time.time() - start_time) / self.steps_per_checkpoint
        loss = step_loss / self.steps_per_checkpoint
        return step_time, loss

    def train(self):
        dataset = self.data_source.get_batch()
        train_bucket_sizes = [len(dataset[b]) for b in range(len(self.data_source.buckets))]
        train_total_size = float(sum(train_bucket_sizes))

        # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
        # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
        # the size if i-th training bucket, as used later.
        self.train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                               for i in range(len(train_bucket_sizes))]
        previous_losses = []
        step_time, loss = 0.0, 0.0
        
        for i in range(self.num_batches):
            print("training batch {}".format(i))
            t, l = self.train_batch(dataset, None)
            step_time += t
            loss += l
            if i % self.steps_per_checkpoint == 0:
                # Print statistics for the previous epoch.
                perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
                print("global step %d learning rate %.4f step-time %.2f perplexity "
                       "%.2f" % (self.global_step.eval(), self.model.learning_rate.eval(),
                                 step_time, perplexity))
                # Decrease learning rate if no improvement was seen over last 3 times.
                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    self.sess.run(self.model.learning_rate_decay_op)
                previous_losses.append(loss)
                # Save checkpoint and zero timer and loss.
                self.save()
                step_time, loss = 0.0, 0.0
                sys.stdout.flush()
                

    def feed(self, features, labels=None):
        response = []
        for sentence in features:
            token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence), self.data_source.vocabA)
            bucket_id = min([b for b in range(len(self.data_source.buckets))
                             if self.data_source.buckets[b][0] > len(token_ids)])
            # Get a 1-element batch to feed the sentence to the model.
            encoder_inputs, decoder_inputs, target_weights = self.model.get_batch(
                {bucket_id: [(token_ids, [])]}, bucket_id)
            # Get output logits for the sentence.
            _, _, output_logits = self.model.step(self.sess, encoder_inputs, decoder_inputs,
                                             target_weights, bucket_id, True)
            # This is a greedy decoder - outputs are just argmaxes of output_logits.
            print(output_logits)
            outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
            # If there is an EOS symbol in outputs, cut them at that point.
            if data_utils.EOS_ID in outputs:
              outputs = outputs[:outputs.index(data_utils.EOS_ID)]
            response.append(" ".join([tf.compat.as_str(rev_out_vocab[output]) for output in outputs]))
        
