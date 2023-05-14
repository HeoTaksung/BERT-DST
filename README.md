# BERT-DST

  * Dialogue State Tracking

  * Chao, Guan-Lin, and Ian Lane. "Bert-dst: Scalable end-to-end dialogue state tracking with bidirectional encoder representations from transformer." arXiv preprint arXiv:1907.03040 (2019). [[Paper]](https://arxiv.org/abs/1907.03040)

  * Dataset

    * [KLUE-DST](https://klue-benchmark.com/tasks/73/overview/description)

        * domain-key-value => slot-key(domain-key), slot-value(value)
        * slot-key => Class defined in Ontology (45 classes in total)
        * slot-value => Values in the dataset

  * Experiment

    * Classification Module (using [CLS] token)

        * 45 slot-keys defined in KLUE-DST and 5 slot-types(none, dontcare, yes, no, span) defined for experiments
        * When predicted with the rest of the classes except none and span, the slot-value is predicted with the rest of the classes (dontcare, yes, no)
        * When predicted as a class with span, the slot-value is determined by the Span Prediction Module
        * When predicted as a class with none, it means there is no slot-key and no slot-value

    * Span Prediction Module (using sequence tokens)

        * When span is predicted in the Classification Module, the slot-value in the text is predicted through start token and end token
        
    * Result through Validation set
        *  Slot F1: 0.908588
        *  Joint-Goal Accuracy: 0.473283
