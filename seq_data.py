from nest.datasource import DataSource
from pymongo import MongoClient
import data_utils
import os
import tensorflow as tf
from tensorflow.python.platform import gfile

_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]

class SeqData(DataSource):
    def __init__(self, save_dir, source='10.0.2.32', is_local=False):
        super().__init__(source, is_local)
        self.client = MongoClient(source)
        self.corpus = 'cornell-corpus'
        self.col = 'dialogs'
        self.open()
        self.buckets = _buckets
        self.vocabfileA = os.path.join(save_dir, self.corpus + '_vocabfileA')
        self.vocabfileB = os.path.join(save_dir, self.corpus + '_vocabfileB')
        if not os.path.isfile(self.vocabfileA):
            self.create_vocab("A", self.vocabfileA)
        if not os.path.isfile(self.vocabfileB):
            self.create_vocab("B", self.vocabfileB)
        print("initializing vocab")
        self.vocabA, self.vocabA_rev = data_utils.initialize_vocabulary(self.vocabfileA)
        self.vocabB, self.vocabB_rev = data_utils.initialize_vocabulary(self.vocabfileB)
        print("vocab initialized") 
    def create_vocab(self, key, vocabfile):
        print("creating vocab %s" % key)
        vocab = {}
        cursor = self.client[self.corpus][self.col].find()
        for pair in cursor:
            line = tf.compat.as_bytes(pair[key].lower())
            tokens = data_utils.basic_tokenizer(line)
            for w in tokens:
                word = data_utils._DIGIT_RE.sub(b"0", w)
                if word in vocab:
                    vocab[word] += 1
                else:
                    vocab[word] = 1
        vocab_list = data_utils._START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        with gfile.GFile(vocabfile, mode="wb") as v_file:
            for w in vocab_list:
                v_file.write(w + b"\n")                
        
    def get_batch(self, batch_size=None):
        super().get_batch(batch_size)
        data_set = [[] for _ in self.buckets]
        for conv_pair in self.cursor:
            source_bytes = tf.compat.as_bytes(conv_pair["A"].lower())
            target_bytes = tf.compat.as_bytes(conv_pair["B"].lower())
            source = data_utils.sentence_to_token_ids(source_bytes, self.vocabA)
            target = data_utils.sentence_to_token_ids(target_bytes, self.vocabB)
            source_ids = [int(x) for x in source]
            target_ids = [int(x) for x in target]
            target_ids.append(data_utils.EOS_ID)
            for bucket_id, (source_size, target_size) in enumerate(self.buckets):
                if len(source_ids) < source_size and len(target_ids) < target_size:
                    data_set[bucket_id].append([source_ids, target_ids])
                    break       
        return data_set

    def open(self):
        self.cursor = self.client[self.corpus][self.col].find()
    def close(self):
        self.cursor.close()
