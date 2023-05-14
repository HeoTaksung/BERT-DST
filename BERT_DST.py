import tensorflow as tf
from transformers import *

# domain-key-value ==> slot-key(domain-key), slot-value(value)
# slot-key => class defined in ontology
# slot-value => detail information about slot-key
# slot-type => none, dontcare, no, yes, span

def compute_loss(labels, logits):
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)

    active_loss = tf.reshape(labels, (-1,)) != -100
        
    reduced_logits = tf.boolean_mask(tf.reshape(logits, (-1, shape_list(logits)[2])), active_loss)
        
    labels = tf.boolean_mask(tf.reshape(labels, (-1,)), active_loss)
    
    return loss_fn(labels, reduced_logits)

    
class BERT_DST(tf.keras.Model):
    def __init__(self, slot_key_class_num, slot_type_class_num, model_name):
        super(BERT_DST, self).__init__()

        self.slot_key_class_num = slot_key_class_num
        self.slot_type_class_num = slot_type_class_num

        self.bert = TFAutoModel.from_pretrained(model_name, output_attentions=True, from_pt=True)

        self.cls_dropout = tf.keras.layers.Dropout(self.bert.config.hidden_dropout_prob)
        self.sequence_dropout = tf.keras.layers.Dropout(self.bert.config.hidden_dropout_prob)

        self.slot_type_classifier = [tf.keras.layers.Dense(slot_type_class_num, kernel_initializer=tf.keras.initializers.TruncatedNormal(self.bert.config.initializer_range)) for i in range(slot_key_class_num)]

        self.span_classifier = [tf.keras.layers.Dense(2, kernel_initializer=tf.keras.initializers.TruncatedNormal(self.bert.config.initializer_range)) for _ in range(slot_key_class_num)]


    def call(self, input_ids, attention_mask=None, token_type_ids=None, training=False):

        bert_outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        cls_token = bert_outputs[0][:, 0, :]
        sequence_tokens = bert_outputs[0]

        pooled_output = self.cls_dropout(cls_token)
        sequence_output = self.sequence_dropout(sequence_tokens)

        slot_type_li = []

        for i in range(self.slot_key_class_num):
            slot_type_li.append(self.slot_type_classifier[i](pooled_output))

        slot_type_output = tf.stack(slot_type_li, axis=1)

        start_li = []
        end_li = []
        
        for i in range(self.slot_key_class_num):
            logits = self.span_classifier[i](sequence_output)
            start_logits, end_logits = tf.split(logits, 2, axis=-1)
            
            start_logits = tf.squeeze(start_logits, axis=-1)
            end_logits = tf.squeeze(end_logits, axis=-1)

            start_li.append(tf.keras.layers.Activation(tf.keras.activations.softmax)(start_logits))
            end_li.append(tf.keras.layers.Activation(tf.keras.activations.softmax)(end_logits))
        
        start_probs = tf.stack(start_li, axis=1)
        end_probs = tf.stack(end_li, axis=1)

        return slot_type_output, start_probs, end_probs
