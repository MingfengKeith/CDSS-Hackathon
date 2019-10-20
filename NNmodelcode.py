
class ReadData():
    """
    Read data
    """
    def __init__(self, dir, name, shuffle=None, mask=-1, class_num=0, batch_size=0):
        assert name in ['train', 'val'], name
        assert os.path.isdir(dir), dir
        cache_file = os.path.join(dir, ".pkl")
        self.name = name
        if shuffle is None:
            shuffle = name == 'train'
        self.shuffle = shuffle
        self.mask = mask
        with open(cache_file.replace(".pkl", name+".pkl") ,"rb") as infile:
            self.data, self.label= pickle.load(infile)

    def size(self):
        return (self.data.shape[0])

    def get_data(self):
        idxs = np.arange(self.data.shape[0])
        if self.shuffle:
            self.rng.shuffle(idxs)
        sample_per_cls_list = np.zeros(self.class_num)
        for k in idxs:
            data = self.data[k]
            if self.mask >= 0:
                data = np.delete(data, self.mask)
            label = self.label[k]
            if self.class_num != 0 and sample_per_cls_list[label] >= self.max_per_cls:
                continue
            if self.class_num != 0:
                sample_per_cls_list[label] += 1
                if np.sum(sample_per_cls_list) >= 100:
                    sample_per_cls_list = np.zeros(self.class_num)
            # label = np.expand_dims(label, -1)
            yield [data, label]

# Network

FEATUREDIM = 6     #  feature dimension
N_CLASSES = 6

def inputs():
    return [tf.placeholder(tf.float32, [None, FEATUREDIM], 'data'),   # bxmaxseqx39
            tf.placeholder(tf.float32, [None], 'label'),  # label is b x maxlen, sparse
            ]

def build_graph(data, label):

    feat = slim.layers.fully_connected(data, 10, scope='fc0')
    feat = slim.layers.fully_connected(feat, 20, scope='fc1')
    feat = slim.layers.fully_connected(feat, 10, scope='fc2')
    feat = slim.layers.fully_connected(feat, 20, scope='fc3')
    logits = slim.layers.fully_connected(feat, N_CLASSES, activation_fn=None, scope="fc4")
    logits = tf.identity(logits, name="logits")
    label = tf.cast(label, tf.int64)
    
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits = logits, labels = label)
    cls_probs = tf.nn.softmax(logits, axis=-1, name="cls_probs")
    cls_preds = tf.argmax(cls_probs, axis=-1, name="cls_preds")
    loss = tf.reduce_sum(loss, name="loss")
    acc = tf.reduce_mean(tf.to_float(tf.equal(cls_preds, label)),
        name='acc')
    return loss


def optimizer():
    lr = tf.get_variable('learning_rate', initializer=0.0, trainable=False)
    return tf.train.GradientDescentOptimizer(lr)


