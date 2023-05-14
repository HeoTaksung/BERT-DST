import collections

def compute_joint_goal_accuracy(true_states, pred_states):
    jga = 0
    jga_len = 0
    for idx, (true_state, pred_state) in enumerate(zip(true_states, pred_states)):
        for true, pred in zip(true_state, pred_state):
            if sorted(true.items()) == sorted(pred.items()):
                jga += 1
            jga_len += 1

    return jga/jga_len

def slot_f1(list_ref, list_hyp):
    F1Scores = collections.namedtuple("F1Scores", ["f1", "precision", "recall"])
    ref = collections.Counter(list_ref)
    hyp = collections.Counter(list_hyp)
    true = sum(ref.values())
    positive = sum(hyp.values())
    true_positive = sum((ref & hyp).values())
    precision = float(true_positive) / positive if positive else 1.0
    recall = float(true_positive) / true if true else 1.0
    if precision + recall > 0.0:
        f1 = 2.0 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return F1Scores(f1=f1, precision=precision, recall=recall)


def compute_slot_f1(true_states, pred_states):
    f1_score_micro = []

    for idx, (true_state, pred_state) in enumerate(zip(true_states, pred_states)):
        for true, pred in zip(true_state, pred_state):
            t = sorted(true.items())
            p = sorted(pred.items())

            true_pred_vocab = set()
            true_pred_vocab.update(t)
            true_pred_vocab.update(p)

            true_pred_index = {i:j for j, i in enumerate(list(true_pred_vocab))}

            t_list = []
            p_list = []
            for t_, p_ in zip(t, p):
                t_list.append(true_pred_index[t_])
                p_list.append(true_pred_index[p_])
                
            f1_score_micro.append(slot_f1(t_list,p_list).f1)

    return sum(f1_score_micro)/len(f1_score_micro)