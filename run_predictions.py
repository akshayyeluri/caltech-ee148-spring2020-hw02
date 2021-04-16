import os
import json
import numpy as np
import json
import pickle
import cv2
from PIL import Image

# Note that you are not allowed to use test data for training.
# set the path to the downloaded data:
data_path = '../RedLights2011_Medium'

# load splits: 
split_path = './hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# set a path for saving predictions:
preds_path = './hw02_preds'
os.makedirs(preds_path, exist_ok=True) # create directory if needed

# Set this parameter to True when you're done with algorithm development:
done_tweaking = True


with open('filter.pkl', 'rb') as f:
    filt = pickle.load(f)

def compute_convolution(I, T, stride=(1, 1), pad=None, 
                        transform = None):
    '''
    This function takes an image <I> and a template <T> (both numpy arrays) 
    and returns a heatmap where each grid represents the output produced by 
    convolution at each location. You can add optional parameters (e.g. stride, 
    window_size, padding) to create additional functionality. 

    transform is a function to be applied before convolution with the filter,
    e.g. normalization of the slice of the image
    '''
    (n_rows,n_cols,n_channels) = np.shape(I)
    k1, k2, k3 = T.shape
    assert(k3 == n_channels)


    # Handle padding
    if pad == None:
        I2 = I
        p1, p2 = 0,0

    # Make result of convolution same shape as original image
    elif pad == 'same' and stride == (1,1):
        n1 = n_rows + k1 - 1
        p1 = (n1 - n_rows) // 2
        n2 = n_cols + k2 - 1
        p2 = (n2 - n_cols) // 2
        I2 = np.zeros((n1, n2, n_channels))
        I2[p1:p1+n_rows, p2:p2+n_cols, :] = I

    else:
        p1, p2 = pad
        n1 = n_rows + 2 * p1
        n2 = n_rows + 2 * p2
        I2 = np.zeroes((n1, n2, n_channels))
        I2[p1:(n1-p1), p2:(n-p2), :] = I


    # Do convolution
    n1, n2, n_channels = I2.shape
    s1, s2 = stride
    shape = ((n1 - k1 + s1)//s1, (n2 - k2 + s2)//s2)
    heatmap = np.empty(shape)
    for i in range(0, n1 - k1 + 1, s1):
        for j in range(0, n2 - k2 + 1, s2):
            i2_slice = I2[i:i+k1, j:j+k2, :]
            if transform:
                i2_slice = transform(i2_slice)
            heatmap[i][j] = np.sum(i2_slice * T)

    return heatmap, p1, p2

def keep_best(bboxes, i, j, x, y, score):
    '''
    Ensure we don't have overlapping boxes,
    so only 1 box per red light
    '''
    bbox_copy = bboxes.copy()
    for (i2, j2), score2 in bbox_copy.items():
        if np.abs(i2 - i) <= x and np.abs(j2 - j) <= y:
            # For all boxes intersecting this box,
            # if the score of the old box is better,
            # return and ignore this box
            if (score2 > score):
                return
            # Otherwise the old box sucks, drop it
            else:
                del bboxes[(i2, j2)]
    bboxes[(i, j)] = score


def predict_boxes(heatmap, x=6, y=8, thresh=0.85):
    '''
    This function takes heatmap and returns the bounding boxes and associated
    confidence scores.
    thresh is the threshold to even consider a bounding box, y is the height
    , x is the width
    '''
    m, n = heatmap.shape
    bounding_boxes0 = {}

    for i in range(m):
        for j in range(n):
            T = heatmap[i][j]
            if (T > thresh):
                # Keep only the best bounding box per red light
                keep_best(bounding_boxes0, i, j, x, y, T)

    output  = [[a, b, a + x, b + y, score] 
               for (a,b), score in bounding_boxes0.items()]

    return output


def detect_red_light_mf_v0(I, filt=filt, thresh=0.85):
    '''
    This function takes a numpy array <I> and returns a list <output>.
    The length of <output> is the number of bounding boxes predicted for <I>. 
    Each entry of <output> is a list <[row_TL,col_TL,row_BR,col_BR,score]>. 
    The first four entries are four integers specifying a bounding box 
    (the row and column index of the top left corner and the row and column 
    index of the bottom right corner).
    <score> is a confidence score ranging from 0 to 1. 

    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''

    heatmap, p1, p2 = compute_convolution(I, filt,
                        transform = lambda x: x / np.linalg.norm(x)
                      )
    output = predict_boxes(heatmap, filt.shape[0], filt.shape[1],
                           thresh=thresh)

    for i in range(len(output)):
        assert len(output[i]) == 5
        assert (output[i][4] >= 0.0) and (output[i][4] <= 1.0)

    return output


############################################################
# Version 2, where we identify red circles first, and
# then do matched filtering using the same filter on sections
# of the image near red circles
############################################################

def read_im(fname, data_path=data_path):
    # read image using PIL:
    I = Image.open(os.path.join(data_path,fname))
    # convert to numpy array:
    return np.asarray(I)


def get_circles(file, filepath=data_path, p2=10):
    '''
    This function returns a list of centers and radii
    for red circles in the image given by file
    '''
    font = cv2.FONT_HERSHEY_SIMPLEX
    img = cv2.imread(os.path.join(filepath,file))
    cimg = img
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # ranges we'll consider red
    lower_red1 = np.array([0,100,100])
    upper_red1 = np.array([10,255,255])
    lower_red2 = np.array([160,100,100])
    upper_red2 = np.array([180,255,255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    maskr = cv2.add(mask1, mask2)

    m,n,_ = img.shape

    # hough circle detect
    r_circles = cv2.HoughCircles(maskr, cv2.HOUGH_GRADIENT, 1, 80,
                               param1=50, param2=p2, minRadius=0, maxRadius=30)


    # needs to be in top bound proportion of image, since lights are high
    # usually
    bound = 8.0 / 10

    centers = []
    radii = []

    if r_circles is not None:
        r_circles = np.uint16(np.around(r_circles))

        for (x,y,r) in r_circles[0, :]:
            # Ensure circle we're finding is in the picture
            if x > n or y > m or y > m*bound or r <= 0:
                continue
            centers.append((x,y))
            radii.append(r)

    return centers, radii



def detect_red_light_mf(fname, filt=filt, thresh = 0.7, filepath=data_path, weaken=False):
    '''
    This function takes a filename <fname> and returns a list <bounding_boxes>.
    '''

    bboxes = []
    scores = []
    if weaken:
        p2 = 1
    else:
        p2 = 10
    centers, radii = get_circles(fname, filepath, p2=p2)
    I = read_im(fname, filepath)
    m, n, _ = I.shape

    for (x,y), r in zip(centers, radii):
        # [1, 2] is aspect ratio, 3 is scaling since width of
        # traffic light is about 3 times the radius of the
        # actual red circle
        dims = np.array([1, 2]) * r * 3
        anchor = (x - (dims[0] // 2), y - dims[1] // 4)
        bbox = [anchor[1], anchor[0], anchor[1] + dims[1], anchor[0] + dims[0]]
        bbox = [max(0, bbox[0]), max(0, bbox[1]),
                min(m, bbox[2]), min(n, bbox[3])]

        I_spliced = I[bbox[0]:bbox[2], bbox[1]:bbox[3], :]
        pil = Image.fromarray(I_spliced)
        I2 = np.asarray(pil.resize((filt.shape[1], filt.shape[0])))
        I3 = I2 / np.linalg.norm(I2)
        score = np.sum(I3 * filt) 
        if (score > thresh):
            bboxes.append(bbox)
            scores.append(score)


    # make sure all scores are below 1
    if (len(scores) > 0):
        scores = np.array(scores) / np.max(scores)

    for i in range(len(bboxes)):
        bboxes[i].append(scores[i])
        assert(len(bboxes[i]) == 5)

    return bboxes



if __name__ == '__main__':
    '''
    Make predictions on the training set.
    '''
    preds_train = {}
    for i in range(len(file_names_train)):

        # read image using PIL:
        I = Image.open(os.path.join(data_path,file_names_train[i]))

        # convert to numpy array:
        I = np.asarray(I)

        #bboxes = detect_red_light_mf_v0(I)
        bboxes = detect_red_light_mf(file_names_train[i])

        preds_train[file_names_train[i]] = [[int(n) for n in val] 
                                 for val in bboxes]

    # save preds (overwrites any previous predictions!)
    with open(os.path.join(preds_path,'preds_train.json'),'w') as f:
        json.dump(preds_train,f)

    if done_tweaking:
        '''
        Make predictions on the test set. 
        '''
        preds_test = {}
        for i in range(len(file_names_test)):
            fname = file_names_test[i]

            # read image using PIL:
            I = Image.open(os.path.join(data_path,fname))

            # convert to numpy array:
            I = np.asarray(I)

            #bboxes = detect_red_light_mf_v0(I)
            bboxes = detect_red_light_mf(fname)
            preds_test[fname] = [[int(n) for n in val] 
                                 for val in bboxes]


        # save preds (overwrites any previous predictions!)
        with open(os.path.join(preds_path,'preds_test.json'),'w') as f:
            json.dump(preds_test,f)
