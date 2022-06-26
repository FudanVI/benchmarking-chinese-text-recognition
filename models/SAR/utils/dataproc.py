'''
This code is to provide necessary utilization functions.
'''

import editdistance
import numpy as np
import zhconv

def strQ2B(ustring):
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374):
            inside_code -= 65248
        rstring += chr(inside_code)
    return rstring

def end_cut(indices, char2id, id2char):
    '''
    indices: numpy array or list of character indices
    charid: char to id conversion
    id2char: id to char conversion
    '''
    cut_indices = []
    for id in indices:
        if id != char2id['END']:
            if id != char2id['UNK'] and id != char2id['PAD']:
                cut_indices.append(id2char[id])
        else:
            break
    return ''.join(cut_indices)

def performance_evaluate(pred_choice, target, voc, char2id, id2char, metrics_type):
    '''
    pred_choice: predicted numpy array of [batch_size, seq_len] with index in output_classes
    target: true numpy array of [batch_size, seq_len] with index in output_classes
    voc: vocabular dictionary
    charid: char to id conversion
    id2char: id to char conversion
    metrics_type: evaluation metric name
    '''
    batch_size = target.shape[0]
    predicts = []
    labels = []
    for batch in range(batch_size):
        predict_indices = pred_choice[batch]
        tareget_indices = target[batch]

        predicts.append(end_cut(predict_indices, char2id, id2char))
        labels.append(end_cut(tareget_indices, char2id, id2char))
        predicts = [zhconv.convert(strQ2B(pred), 'zh-cn') for pred in predicts]
        labels = [zhconv.convert(strQ2B(tar), 'zh-cn')for tar in labels]

    if metrics_type == 'accuracy':
        acc_list = [(pred == tar) for pred, tar in zip(predicts, labels)]
        accuracy = 1.0 * sum(acc_list) / len(acc_list)

        return accuracy, acc_list, predicts, labels
    elif metrics_type == 'editdistance':
        ed_list = [editdistance.eval(pred, targ) for pred, targ in zip(predicts, labels)]
        eds = 1.0 * sum(ed_list) / len(ed_list)

        return eds, ed_list, predicts, labels

    return -1

# unit test
if __name__ == '__main__':
    import sys
    sys.path.append("..")

    from dataset.dataset import dictionary_generator

    batch_size = 2
    seq_len = 40
    voc, char2id, id2char = dictionary_generator()
    print("Vocabulary size is:", len(voc))

    pred_choice = np.random.randint(0,len(voc),(batch_size, seq_len)) # [batch_size, seq_len]
    target = np.array([[47, 44, 57, 44, 49, 42, 95, 95, 95, 95, 95, 95, 95, 95, 95, 95, 95, 95,
                        95, 95, 95, 95, 95, 95, 95, 95, 95, 95, 95, 95, 95, 95, 95, 95, 95, 95,
                        95, 95, 95, 94],
                       [54, 55, 36, 49, 39, 36, 53, 39, 95, 95, 95, 95, 95, 95, 95, 95, 95, 95,
                        95, 95, 95, 95, 95, 95, 95, 95, 95, 95, 95, 95, 95, 95, 95, 95, 95, 95,
                        95, 95, 95, 94]])

    word = end_cut(pred_choice[0], char2id, id2char)
    print("First decode word is:", word)
    word = end_cut(pred_choice[1], char2id, id2char)
    print("Second decode word is:", word)

    word = end_cut(target[0], char2id, id2char)
    print("First decode word is:", word)
    word = end_cut(target[1], char2id, id2char)
    print("Second decode word is:", word)

    metric, metric_list, predicts, labels = performance_evaluate(pred_choice, target, voc, char2id, id2char, 'accuracy')
    print("Accuracy:", metric)
    print("Accuracy list:", metric_list)
    print("Predicted words:", predicts)
    print("Labeled words:", labels)
    metric, metric_list, predicts, labels = performance_evaluate(pred_choice, target, voc, char2id, id2char, 'editdistance')
    print("Edit distance:", metric)
    print("Edit distance list:", metric_list)
    print("Predicted words:", predicts)
    print("Labeled words:", labels)