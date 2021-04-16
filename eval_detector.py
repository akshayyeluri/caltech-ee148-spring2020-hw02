import os
import json
import numpy as np
import matplotlib.pyplot as plt

def compute_iou(box_1, box_2):
    '''
    This function takes a pair of bounding boxes and returns intersection-over-
    union (IoU) of two bounding boxes.
    '''
    y1, x1, y2, x2 = box_1
    y3, x3, y4, x4 = box_2

    x_int = min(x2, x4) - max(x1, x3)
    y_int = min(y2, y4) - max(y1, y3)

    if (x_int < 0 or y_int < 0):
        inter = 0
    else:
        inter = x_int * y_int

    # A U B = A + B - A \cap B
    union = (x2 - x1) * (y2 - y1) + (x4 - x3) * (y4 - y3) - inter

    iou = inter / union
    return iou


def compute_counts(preds, gts, iou_thr=0.5, conf_thr=0.5):
    '''
    This function takes a pair of dictionaries (with our JSON format; see ex.) 
    corresponding to predicted and ground truth bounding boxes for a collection
    of images and returns the number of true positives, false positives, and
    false negatives. 
    <preds> is a dictionary containing predicted bounding boxes and confidence
    scores for a collection of images.
    <gts> is a dictionary containing ground truth bounding boxes for a
    collection of images.
    '''
    TP = 0
    FP = 0
    FN = 0

    '''
    BEGIN YOUR CODE
    '''
    for pred_file, pred in preds.items():
        gt = gts[pred_file]

        pred = [p for p in pred if p[4] >= conf_thr]
        matching = set()
        for i in range(len(gt)):
            found_match = False
            for j in range(len(pred)):
                iou = compute_iou(pred[j][:4], gt[i])
                if (iou >= iou_thr):
                    TP += 1
                    matching.add(j)
                    found_match = True
                    break
            if not found_match:
                FN += 1

        for j in range(len(pred)):
            if (j not in matching):
                FP += 1

    return TP, FP, FN

# set a path for predictions and annotations:
preds_path = './hw02_preds'
gts_path = './hw02_annotations'

# load splits:
split_path = './hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# Set this parameter to True when you're done with algorithm development:
done_tweaking = True

'''
Load training data. 
'''
with open(os.path.join(preds_path,'preds_train.json'),'r') as f:
    preds_train = json.load(f)
    
with open(os.path.join(gts_path, 'annotations_train.json'),'r') as f:
    gts_train = json.load(f)


if done_tweaking:
    
    '''
    Load test data.
    '''
    
    with open(os.path.join(preds_path,'preds_test.json'),'r') as f:
        preds_test = json.load(f)
        
    with open(os.path.join(gts_path, 'annotations_test.json'),'r') as f:
        gts_test = json.load(f)



def make_pr_curve(preds, gts, iou_thresh=0.5, confidence_thrs=None):
    tp = np.zeros(len(confidence_thrs))
    fp = np.zeros(len(confidence_thrs))
    fn = np.zeros(len(confidence_thrs))
    for i, conf_thr in enumerate(confidence_thrs):
        tp[i], fp[i], fn[i] = compute_counts(preds, gts, 
                iou_thr=iou_thresh, conf_thr=conf_thr)
    prec = tp / (tp + fp)
    recall = tp / (tp + fn)
    return prec, recall



if __name__ == '__main__':
    # For a fixed IoU threshold, vary the confidence thresholds.
    # The code below gives an example on the training set for one IoU threshold. 

    scores = sum([[bbox[4] for bbox in v] for k,v in preds_train.items()], [])
    # using (ascending) list of confidence scores as thresholds
    confidence_thrs = np.sort(np.array(scores)) 


    # Plot training set PR curves

    if done_tweaking:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, thresh in zip(axes, [0.25, 0.5, 0.75]):
            prec_train, recall_train = make_pr_curve(preds_train, gts_train, thresh, confidence_thrs)
            prec_test, recall_test = make_pr_curve(preds_test, gts_test, thresh, confidence_thrs)
            ax.plot(recall_test, prec_test, label='Test')
            ax.plot(recall_train, prec_train, label='Train')
            ax.legend()
            ax.set_title(f'PR Curve, IoU thresh={thresh}')
            ax.set_xlabel('Recall'); ax.set_ylabel('Precision')

        plt.savefig('pr.png')

