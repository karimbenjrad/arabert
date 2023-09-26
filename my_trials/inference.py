from arabert.modeling import BertModel, BertConfig
import tensorflow as tf

config = BertConfig()
input_ids = tf.constant([[31, 51, 99], [15, 5, 0]])
model = BertModel(config=config, is_training=True,
      input_ids=input_ids)

