# domain-key-value ==> slot-key(domain-key), slot-value(value)
# slot-key => ontology에 정의된 class
# slot-value => slot-key에 대한 상세 정보
# slot-type => none, dontcare, no, yes, span

slot_key_class_num = len(slot_key)
slot_type_class_num - len(slot_type)

with strategy.scope():
    input_ids = tf.keras.layers.Input(shape=(dataset['input_ids'].shape), dtype=tf.int32)
    attention_masks = tf.keras.layers.Input(shape=(dataset['attention_masks'].shape), dtype=tf.int32)
    token_type_ids = tf.keras.layers.Input(shape=(dataset['token_type_ids'].shape), dtype=tf.int32)
    
    bert = TFBertModel.from_pretrained('klue/bert-base', output_attentions=True, from_pt=True)
    
    bert_outputs = bert(input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids)
    
    cls_token = bert_outputs[0][:, 0, :]
    sequence_tokens = bert_outputs[0]

    pooled_output = tf.keras.layers.Dropout(bert.config.hidden_dropout_prob)(cls_token, training=False)
    sequence_output = tf.keras.layers.Dropout(bert.config.hidden_dropout_prob)(sequence_tokens, training=False)
    
    slot_type_classifier = [tf.keras.layers.Dense(slot_type_class_num, kernel_initializer=tf.keras.initializers.TruncatedNormal(bert.config.initializer_range)) for i in range(slot_key_class_num)]
    
    slot_type_li = []

    for i in range(slot_key_class_num):
        slot_type_li.append(slot_type_classifier[i](pooled_output))

    slot_type_output = tf.stack(token_type_li, axis=1)
    slot_type_output = tf.keras.layers.Reshape((slot_key_class_num, slot_type_class_num), name='key_type_outputs')(slot_type_output)
    
    span_classifier = [tf.keras.layers.Dense(2, kernel_initializer=tf.keras.initializers.TruncatedNormal(bert.config.initializer_range)) for _ in range(slot_key_class_num)]
    
    start_li = []
    end_li = []
    
    for i in range(slot_key_class_num):
        logits = span_layers[i](sequence_output)
        start_logits, end_logits = tf.split(logits, 2, axis=-1)
        
        start_logits = tf.squeeze(start_logits, axis=-1)
        end_logits = tf.squeeze(end_logits, axis=-1)

        start_li.append(tf.keras.layers.Activation(tf.keras.activations.softmax)(start_logits))
        end_li.append(tf.keras.layers.Activation(tf.keras.activations.softmax)(end_logits))
    
    start_li = tf.stack(start_li, axis=1)
    end_li = tf.stack(end_li, axis=1)
    
    start_probs = tf.keras.layers.Reshape((slot_key_class_num, 512,), name='start_output')(start_li)
    end_probs = tf.keras.layers.Reshape((slot_key_class_num, 512,), name='end_output')(end_li)
    
    model = tf.keras.Model(inputs=[input_ids, attention_masks, segment_ids], outputs=[slot_type_output, start_probs, end_probs])
    
    optimizer = tf.keras.optimizers.Adam(1e-5)
    loss_1 = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    loss_2 = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    loss_3 = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    model.compile(optimizer=optimizer, loss={'key_type_outputs':loss_1, 'start_output':loss_2, 'end_output':loss_3})
