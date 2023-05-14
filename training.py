from utils import *
import os
import tensorflow as tf
from transformers import *
from tensorflow.keras.callbacks import EarlyStopping
from silence_tensorflow import silence_tensorflow
from BERT_DST import compute_loss, BERT_DST
from eval_matrix import compute_joint_goal_accuracy, compute_slot_f1

os.environ["CUDA_VISIBLE_DEVICES"]="0, 1, 2, 3"

strategy = tf.distribute.MirroredStrategy(devices=["GPU:0", "GPU:1", "GPU:2", "GPU:3"])

silence_tensorflow()

train_dir = "/home/hts221/DST/data/wos-v1.1/wos-v1.1_train.json"
dev_dir = "/home/hts221/DST/data/wos-v1.1/wos-v1.1_dev.json"
ontology_dir = "/home/hts221/DST/data/wos-v1.1/ontology.json"
tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
max_length = 512

train_dataset = DSTDataset(data_dir=train_dir, ontology_dir=ontology_dir, tokenizer=tokenizer, max_length=max_length)
dev_dataset = DSTDataset(data_dir=dev_dir, ontology_dir=ontology_dir, tokenizer=tokenizer, max_length=max_length)

train_inputs = model_input_change(train_dataset)
dev_inputs = model_input_change(dev_dataset)

slot_key_class_num = len(train_dataset.get_vocab()[0])
slot_type_class_num = len(train_dataset.get_vocab()[1])

with strategy.scope():
    model = BERT_DST(slot_key_class_num, slot_type_class_num, model_name="klue/bert-base")
    optimizer = tf.keras.optimizers.Adam(3e-5)
    loss_1 = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss={'output_1':loss_1, 'output_2':compute_loss, 'output_3':compute_loss})

earlystop_callback = EarlyStopping(monitor='val_output_1_loss', verbose=1, min_delta=0.0001, patience=2, mode='min', restore_best_weights=True)

model.fit([train_inputs['input_ids'], train_inputs['attention_mask'], train_inputs['token_type_ids']], {'output_1':train_inputs['slot_type_labels'], 'output_2':train_inputs['start_index'], 'output_3':train_inputs['end_index']}, epochs=60, batch_size=16, 
                    validation_data=([dev_inputs['input_ids'], dev_inputs['attention_mask'], dev_inputs['token_type_ids']], {'output_1':dev_inputs['slot_type_labels'], 'output_2':dev_inputs['start_index'], 'output_3':dev_inputs['end_index']}), callbacks=[earlystop_callback])

y_pred = model.predict([dev_inputs['input_ids'], dev_inputs['attention_mask'], dev_inputs['token_type_ids']], batch_size=16, verbose=1)

slot_type_pred = y_pred[0]
start_pred = y_pred[1]
end_pred = y_pred[2]

idx_to_slot_type = {i: j for i, j in enumerate(train_dataset.get_vocab()[1])}
idx_to_slot_key = {i: j for i, j in enumerate(train_dataset.get_vocab()[0])}


true_states, pred_states = model_output_change(idx_to_slot_type, idx_to_slot_key, dev_inputs, slot_type_pred, start_pred, end_pred, dev_dir, tokenizer)

print("JGA Score: ", compute_joint_goal_accuracy(true_states, pred_states))
print("Slot F1 Score : ", compute_slot_f1(true_states, pred_states))