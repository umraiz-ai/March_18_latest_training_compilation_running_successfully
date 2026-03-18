# Copyright 2023 The Deeper-I Authors. All Rights Reserved.
#
# BSD 3-Clause License
#
# Copyright (c) 2017,
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import numpy as np
# np.seterr(all='print')

EPSILON_16 = 9.77e-04
EPSILON_32 = 1.19e-07
EPSILON_64 = 2.22e-16
EPSILON = EPSILON_32

def sine(x):
    return np.sin(x)

def exp(x):
    return np.exp(x)

def identity_function(x):
    return x


def step_function(x):
    return np.array(x > 0, dtype=np.int)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))    

def tabled_sigmoid(x, sig_table):
    x = (x.reshape(-1)*256).astype(np.int32) + 2048
    x[x>4095] = 4095
    x[x<0] = 0
    return sig_table[x]


def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)
    

def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    grad = np.zeros(x)
    grad[x>=0] = 1
    return grad
    

def softmax(x):
    x_nor = x - np.max(x, axis=-1)[...,None]
    x_sum = np.sum(np.exp(x_nor), axis=-1)[...,None]
    o = np.exp(x_nor) / (x_sum + EPSILON)
    return o


def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size


def softmax_loss(X, t):
    y = softmax(X)
    return cross_entropy_error(y, t)

class EMA:
    '''
        input     : tuple()
        operation : list()
        output    : tuple()
    '''
    def __init__(self, points=[], alpha=0.4, margin=150):
        self.points = points
        self.alpha = alpha
        self.margin = margin

    def calc(self, points):
        newpoints = [] #volatile

        if self.points == []: # initial condition
            self.points = list(points)

        for i in range(len(points)):
            if abs(self.points[i] - points[i])>= self.margin: # long distance condition
                self.points[i] = int(0.7*points[i] + self.points[i]*0.3)

            one = int(self.points[i]*(1-self.alpha) + self.alpha*points[i])
            newpoints.append(one)

        self.points = newpoints
        # self.points = newpoints.copy()
        return tuple(newpoints)


def _get_padding(
    input_size, 
    kernel_size, 
    output_size, 
    stride,
    pad_order='down_right',
):
    # pad = ((stride*(output_size-1))+kernel_size-input_size)/2.
    # pad = ((stride*(output_size-1))+kernel_size-input_size)*1.0/2.0
    pad = float((stride*(output_size-1))+kernel_size-input_size)
    if pad < 0.: pad = 0.
    if pad_order == 'down_right':
        pads = (int(pad // 2), int(np.ceil(pad / 2)))
    else:
        pads = (int(np.ceil(pad / 2)), int(pad // 2))

    return pads

def _get_output_valid(xsize, ksize, stride):
    if stride == 1:
        out = xsize - ksize + 1
    else:
        xsize = np.max((1, xsize - ksize))
        out = xsize // stride + 1

    return out

def get_info(h, w, fh, fw, pad_mode, stride_h, stride_w, pad_order='down_right'):
    if pad_mode == 'SAME':
        # out_h = int(H*1.0 / stride_h)  # round down
        # out_w = int(W*1.0 / stride_w)  # round down
        out_h = int(h * 1.0 / stride_h + 0.5)  # round 
        out_w = int(w * 1.0 / stride_w + 0.5)  # round 
        pad_h = _get_padding(h, fh, out_h, stride_h, pad_order=pad_order)
        pad_w = _get_padding(w, fw, out_w, stride_w, pad_order=pad_order)
        pads = (pad_h, pad_w)
        
    elif pad_mode == 'VALID':
        out_h = _get_output_valid(h, fh, stride_h)
        out_w = _get_output_valid(w, fw, stride_w)
        pads = ((0, 0), (0, 0))

    elif pad_mode == 'TRANSPOSE':
        out_h = int(h * 1.0 / stride_h + 0.5)  # round 
        out_w = int(w * 1.0 / stride_w + 0.5)  # round 
        pad_h = _get_padding(h, fh, out_h, stride_h, pad_order=pad_order)
        pad_w = _get_padding(w, fw, out_w, stride_w, pad_order=pad_order)
        pads = ((pad_h[0] + pad_h[1], 0), (pad_w[0] + pad_w[1], 0))

    else:  # Default 'SAME'
        out_h = int(h * 1.0 / stride_h)  # round down
        out_w = int(w * 1.0 / stride_w)  # round down
        pad_h = _get_padding(h, fh, out_h, stride_h, pad_order=pad_order)
        pad_w = _get_padding(w, fw, out_w, stride_w, pad_order=pad_order)
        pads = (pad_h, pad_w)

    return pads, out_h, out_w


def smooth_curve(x):
    window_len = 11
    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    w = np.kaiser(window_len, 2)
    y = np.convolve(w/w.sum(), s, mode='valid')
    return y[5:len(y)-5]


def shuffle_dataset(x, t):
    permutation = np.random.permutation(x.shape[0])
    x = x[permutation,:] if x.ndim == 2 else x[permutation,:,:,:]
    t = t[permutation]

    return x, t


def conv_output_size(input_size, filter_size, stride=1, pad=0):
    return (input_size + 2*pad - filter_size) / stride + 1


def im2col(
    input_data, 
    filter_h, filter_w, 
    out_h, out_w, 
    stride_h=1, stride_w=1, 
    pads=((0,0), (0,0)), 
    pad_value=0, 
    dtype="float32", 
):
    N, H, W, C = input_data.shape
    image = input_data.transpose(0, 3, 1, 2)

    pad_up_down, pad_left_right = pads

    img = np.pad(
        image, 
        [(0, 0), (0, 0), pad_up_down, pad_left_right], 
        'constant', 
        constant_values=pad_value
    )
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w), dtype=dtype)

    for y in range(filter_h):
        y_max = y + stride_h * out_h
        for x in range(filter_w):
            x_max = x + stride_w * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride_h, x:x_max:stride_w]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)

    return col


def im2col_nhwc(input_data, filter_h, filter_w, stride=1, pad=0):
    N, H, W, C = input_data.shape
    image = input_data.transpose(0, 3, 1, 2)
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = np.pad(image, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            # print('col {}'.format(col.shape))
            # print('img {}'.format(img.shape))
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col


def im2col_nchw(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col

'''
def im2col_nchw(input_data, filter_h, filter_w, out_h, out_w, stride=1, pad=(0.0,0.0), dtype=np.float16, pad_value=0):
    N, C, H, W = input_data.shape

    pad_up_down = (int(pad[0]), int(pad[0]+0.5))
    pad_left_right = (int(pad[1]), int(pad[1]+0.5))

    img = np.pad(input_data, [(0, 0), (0, 0), pad_up_down, pad_left_right], 'constant', constant_values=pad_value)
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)

    return col
'''

# def col2im(col, input_shape, filter_h, filter_w, stride_h=1, stride_w=1, pad=(0.0,0.0), dtype=np.float16):
def col2im(col, input_shape, filter_h, filter_w, stride_h=1, stride_w=1, pad=(0.0,0.0), dtype="float32"):
    N, C, H, W = input_shape
    out_h = (H + 2*pad[0] - filter_h)//stride_h + 1
    out_w = (W + 2*pad[1] - filter_w)//stride_w + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad[0] + stride_h - 1, W + 2*pad[1] + stride_w - 1), dtype=dtype)
    for y in range(filter_h):
        y_max = y + stride_h*out_h
        for x in range(filter_w):
            x_max = x + stride_w*out_w
            img[:, :, y:y_max:stride_h, x:x_max:stride_w] += col[:, :, y, x, :, :]

    return img[:, :, pad[0]:H + pad[0], pad[1]:W + pad[1]]


def transpose_kernel(w, transpose=False):
    if transpose: 
        w = np.transpose(w, (0,1,3,2))
        w = w[::-1, ::-1, : , :]
    H, W, I, O = w.shape
    w_2d = w.transpose(2, 0, 1, 3).reshape(I*H*W, O)

    return w_2d


def get_category(x, mode='TOP1'):
    """
    Get category from output of fc1000.
    :param x: narray (1,1000), output of fc1000 (ImageNet)
    :param mode: string / 'TOP1' or 'TOP5'
    :return: list, list / index, value
    """

    index_list = []
    value_list = []

    cnt = 1
    if mode == 'TOP1':
        cnt = 1
    elif mode == 'TOP5':
        cnt = 5

    for i in range(cnt):
        index = np.argmax(x)
        value = np.max(x)
        x = np.delete(x, index)

        index_list.append(index)
        value_list.append(value)

    return index_list, value_list
    # return result_i, result_v


def dump_data(x, name, output_path):
    if x.ndim == 4:
        N, H, W, C = x.shape
        x_2d = x.transpose(0, 1, 3, 2).reshape(N*H*C, W)

        def x_rename(type): return '%s/%s_%d_%d_%d_%d.%s' % (output_path, name, N, H, W, C, type)

        x_2d.byteswap().tofile(x_rename('bin'))
        # mem_drv.write_to_file(x_2d, x_rename('bin'), dtype=dtype)
        np.savetxt(x_rename('txt'), x_2d, delimiter=',', fmt='%1.6f')

    else: # x.ndim == 1

        def x_rename(type):
            return '%s/%s_%d.%s' % (output_path, name, x.shape[1], type)

        x.byteswap().tofile(x_rename('bin'))
        # mem_drv.write_to_file(x, x_rename('bin'), dtype=dtype)
        np.savetxt(x_rename('txt'), x, delimiter=',', fmt='%1.6f')


def save_image_as_bin(x, name, output_path):
    N, H, W, C = x.shape
    x_2d = x.transpose(0, 1, 3, 2).reshape(N*H*C, W)

    def x_rename(type): return '%s/%s.%s' % (output_path, name, type)

    x_2d.astype(dtype=np.float16).byteswap().tofile(x_rename('bin'))
    # x_2d.byteswap().tofile(x_rename('bin'))


def generate_bbox(cls_map, reg, scale, threshold, stride=2, cellsize=12, dtype=np.float32):

    t_index = np.where(cls_map > threshold)

    # find nothing
    if t_index[0].size == 0:
        return np.array([], dtype=dtype)

    #offset
    dx1, dy1, dx2, dy2 = [reg[t_index[0], t_index[1], i] for i in range(4)]

    reg = np.array([dx1, dy1, dx2, dy2], dtype=dtype)
    score = cls_map[t_index[0], t_index[1]]
    np.set_printoptions(threshold=np.inf)
    boundingbox = np.vstack([np.round((stride * t_index[1]) / scale),
                             np.round((stride * t_index[0]) / scale),
                             np.round((stride * t_index[1] + cellsize) / scale),
                             np.round((stride * t_index[0] + cellsize) / scale),
                             score,
                             reg])
    # print('boundingbox')
    # print(boundingbox)
    
    return boundingbox.T


def convert_to_square(bbox):
    square_bbox = bbox

    h = bbox[:, 3] - bbox[:, 1] + 1
    w = bbox[:, 2] - bbox[:, 0] + 1
    max_side = np.maximum(h, w)
    square_bbox[:, 0] = bbox[:, 0] + w * 0.5 - max_side * 0.5
    square_bbox[:, 1] = bbox[:, 1] + h * 0.5 - max_side * 0.5
    square_bbox[:, 2] = square_bbox[:, 0] + max_side - 1
    square_bbox[:, 3] = square_bbox[:, 1] + max_side - 1

    return square_bbox


def pad(bboxes, w, h, dtype=np.float32):
    tmpw, tmph = bboxes[:, 2] - bboxes[:, 0] + 1, bboxes[:, 3] - bboxes[:, 1] + 1
    num_box = bboxes.shape[0]

    dx, dy = np.zeros((num_box,), dtype=dtype), np.zeros((num_box,), dtype=dtype)
    edx, edy = tmpw - 1, tmph - 1

    x, y, ex, ey = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]

    tmp_index = np.where(ex > w - 1)
    edx[tmp_index] = tmpw[tmp_index] + w - 2 - ex[tmp_index]
    ex[tmp_index] = w - 1

    tmp_index = np.where(ey > h - 1)
    edy[tmp_index] = tmph[tmp_index] + h - 2 - ey[tmp_index]
    ey[tmp_index] = h - 1

    tmp_index = np.where(x < 0)
    dx[tmp_index] = 0 - x[tmp_index]
    x[tmp_index] = 0

    tmp_index = np.where(y < 0)
    dy[tmp_index] = 0 - y[tmp_index]
    y[tmp_index] = 0

    return_list = [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph]
    return_list = [item.astype(np.int32) for item in return_list]

    return return_list


def calibrate_box(bbox, reg):
    bbox_c = bbox
    w = bbox[:, 2] - bbox[:, 0] + 1
    h = bbox[:, 3] - bbox[:, 1] + 1
    reg_m = np.hstack([w, h, w, h]).reshape(-1, 4)
    aug = reg_m * reg
    bbox_c[:, 0:4] = bbox_c[:, 0:4] + aug

    return bbox_c


def calibrate_landmark(bbox, reg):
    points = reg

    w = bbox[:, 2] - bbox[:, 0] + 1
    h = bbox[:, 3] - bbox[:, 1] + 1
    reg_m = np.hstack([w, h]).reshape(-1, 2)

    # points[:, 0:2] = bbox[:, :2] + (reg_m * points[:, 0:2])
    # points[:, 2:4] = bbox[:, :2] + (reg_m * points[:, 2:4])
    # points[:, 4:6] = bbox[:, :2] + (reg_m * points[:, 4:6])
    # points[:, 6:8] = bbox[:, :2] + (reg_m * points[:, 6:8])
    # points[:, 8:10] = bbox[:, :2] + (reg_m * points[:, 8:10])
    # points[:, 10:12] = bbox[:, :2] + (reg_m * points[:, 10:12])

    # points[:, 0::2] = (np.tile(w, (6, 1)) * points[:, 0::2].T + np.tile(bbox[:, 0], (6, 1)) - 1).T
    # points[:, 1::2] = (np.tile(h, (6, 1)) * points[:, 1::2].T + np.tile(bbox[:, 1], (6, 1)) - 1).T

    points[:, 0::2] = (w * points[:, 0::2].T + (bbox[:, 0])).T
    points[:, 1::2] = (h * points[:, 1::2].T + (bbox[:, 1])).T

    return points


def py_nms(cls, dets, thresh, mode="Union", score_index=1):
    """
    greedily select boxes with high confidence
    keep boxes overlap <= thresh
    rule out overlap > thresh
    :param dets: [[x1, y1, x2, y2 score]]
    :param thresh: retain overlap <= thresh
    :return: indexes to keep
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = cls[:, score_index]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if mode == "Union":
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
        elif mode == "Minimum":
            ovr = inter / np.minimum(areas[i], areas[order[1:]])

        #keep
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def IoU(box, bboxes):
    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    areas = (bboxes[:, 2] - bboxes[:, 0] + 1) * (bboxes[:, 3] - bboxes[:, 1] + 1)
    xx1 = np.maximum(box[0], bboxes[:, 0])
    yy1 = np.maximum(box[1], bboxes[:, 1])
    xx2 = np.minimum(box[2], bboxes[:, 2])
    yy2 = np.minimum(box[3], bboxes[:, 3])

    # compute the width and height of the bounding box
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    inter = w * h
    over = inter / (box_area + areas - inter)

    return over


def load_params(param_dir, verbose=False, dtype=np.float32):
    params = {}
    # Load npy files
    # todo: initialize params (for train)
    files = glob.glob(param_dir + '/*.npy')
    
    start_time = time.time()

    for file in files:
        # print(file)
        f = np.load(file.strip())
        name = os.path.basename(file).split('.')[0]
        # params[name] = f.astype(dtype=dtype)
        params[name] = f.astype(dtype=np.float16)
        params[name] = params[name].astype(dtype=dtype)
    
    elapsed_time = time.time() - start_time
    if verbose:
        print('[INFO]: Parameters Loading Time=%s sec'%elapsed_time)

    return params


def lcnn_load_params(weight_dir, dtype=np.float32):
    params = {}
    files = glob.glob(weight_dir + "/*.npy")

    for file in files:
        # print(file)
        f = np.load(file.strip())
        layer, wtype = os.path.basename(file).split('_')[:2]
        params[layer + '_' + wtype] = f.astype(dtype)

    return params

def preprocessor(image, crop=None, output_size=None, toGray=False, normalize=False, mean=np.array([127.5]), std=np.array([128.0])):
    h, w, c = image.shape
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if toGray is True and c == 3 else image
    img = img[crop[1]:crop[3], crop[0]:crop[2]] if crop is not None else img
    img = cv2.resize(img, (output_size[1], output_size[0]), interpolation=cv2.INTER_NEAREST) if output_size is not None else img
    img = (img - mean) / std if normalize is True else img
    img = img[:, :, np.newaxis] if len(img.shape) != 3 else img
 
    return img

def lcnn_get_distance(self, embedding1, embedding2, dist_metric):
    if dist_metric==0: # Original Cosine Similarity
        # py2 dependency
        #dist = 1 - np.dot(embedding1.T, embedding2) / (np.linalg.norm(embedding1)*np.linalg.norm(embedding2)+1e-5)
        dot = np.sum(np.multiply(embedding1, embedding2))
        norm = (np.linalg.norm(embedding1)*np.linalg.norm(embedding2)+1e-5)
        dist = (dot/norm)      # 1, the same person is recognized
        #dist = 1 - (dot/norm) # 0, the same person is recognized

    elif dist_metric==1: # Custom Cosine Simility
        dot  = np.sum(np.multiply(embedding1, embedding2))
        norm = (np.linalg.norm(embedding1)*np.linalg.norm(embedding2)+1e-5)
        similarity = dot / norm
        #dist = (np.arccos(similarity) / math.pi)    # 0, the same person is recognized
        dist = 1 - (np.arccos(similarity) / math.pi) # 1, the same person is recognized
    else:
        raise 'Undefined distance metric %d' % dist_metric

    return dist

def lcnn_save_result(x, name, output_dir='lcnn_results/'):
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    ndim = x.ndim

    if ndim >= 3:
        n, c, h, w = np.shape(x)
        x2d = np.reshape(x, (n * c, h * w))

    elif ndim < 3:
        x2d = x

    # print("save layer out of", name)
    np.savetxt(output_dir + name + '.txt', x2d, fmt="%f")


def dump_json_and_img(img, size=[], output_dir='.', car=[0,0,0,0], plate=[0,0,0,0], ocr=''):
    result = {}
    in_h, in_w = size
    now = datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S%f')
    result['timestamp'] = now
    factor = 1000.0

    x1, y1, x2, y2 = car
    x1=max(0, x1); x2=min(in_w, x2); y1=max(0, y1); y2=min(in_h, y2); 
    w = x2-x1 if x2-x1>0 else 0; h = y2-y1 if y2-y1>0 else 0
    result['car'] = {'x':int(x1*factor/in_w), 'y':int(y1*factor/in_h), 'w':int(w*factor/in_w), 'h':int(h*factor/in_h)} 

    x1, y1, x2, y2 = plate; 
    x1=max(0, x1); x2=min(in_w, x2); y1=max(0, y1); y2=min(in_h, y2); 
    w = x2-x1 if x2-x1>0 else 0; h = y2-y1 if y2-y1>0 else 0
    result['plate'] = {'x':int(x1*factor/in_w), 'y':int(y1*factor/in_h), 'w':int(w*factor/in_w), 'h':int(h*factor/in_h)} 

    result['ocr'] = {'data':ocr} 

    with open('{}.json'.format(os.path.join(output_dir, now)), 'w') as json_file:
        json.dump(result, json_file)

    # cv2.imshow('', img)
    # cv2.waitKey(0)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV_I420)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV_I420)
    with open('{}.yuv420'.format(os.path.join(output_dir, now)), 'wb') as img_file:
        img.tofile(img_file)


def get_biggest_face(boxes_c, landmarks):
    bface = 0
    bid = None

    for i in range(boxes_c.shape[0]):
        bbox = boxes_c[i, :4]
        candidate = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

        if candidate > bface:
            bface = candidate
            bid = i

        bbboxes = boxes_c[bid][:4].astype(np.int)
        bscore = boxes_c[bid][4]
        blandmarks = landmarks[bid][:].astype(np.int)

    # for i in range(landmarks.shape[0]):

    return bbboxes, blandmarks, bscore


def draw(image, landmark, bbox, score, p1, p2):
    #cv2.circle(image, (landmark[ 0], landmark[ 1]), 5, (255,0,0)) # l-eye
    #cv2.circle(image, (landmark[ 2], landmark[ 3]), 5, (255,0,0)) # r-eye
    cv2.circle(image, (landmark[ 4], landmark[ 5]), 5, (255,0,0)) # nose
    #cv2.circle(image, (landmark[ 6], landmark[ 7]), 5, (255,0,0)) # l-mouse
    #cv2.circle(image, (landmark[ 8], landmark[ 9]), 5, (255,0,0)) # r-mouse
    #cv2.circle(image, (landmark[10], landmark[11]), 5, (255,0,0)) # chin

    cv2.line(image, p1, p2, (0, 255, 0), 2)

    ## cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 1)
    # cv2.putText(image, '{:.3f}'.format(score), (bbox[0], bbox[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    # cv2.putText(image, '{:d}X{:d}'.format(360, 360), (10, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)


def get_camera_matrix(image):
    size = image.shape
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_metrix = np.array([[focal_length, 0, center[0]],
                              [0, focal_length, center[1]],
                              [0, 0, 1]], dtype = "float")
    return camera_metrix


def get_model_points():
    # good to rnet
    model_points = np.array([   (   0.0,    0.0,    0.0),   # Nose tip
                                (   0.0, -330.0,  -65.0),   # Chin
                                (-160.0,  180.0, -160.0),   # Left eye left corner
                                ( 160.0,  180.0, -160.0),   # Right eye right corne
                                (-150.0, -150.0, -150.0),   # Left Mouth corner
                                ( 150.0, -150.0, -150.0)])  # Right mouth corner
    return model_points


def get_default_image_points():
    image_points = np.array([(0, 0),     # Nose tip
                             (0, 0),     # chin
                             (0, 0),     # Left eye left corner
                             (0, 0),     # Right eye right corne
                             (0, 0),     # Left Mouth corner
                             (0, 0)      # Right mouth corner
                            ], dtype="float")
    return image_points


def head_gaze_estimation(landmark, camera_matrix, model_points, image_points):
    # - Assuming no lens distortion
    dist_coeffs = np.zeros((4, 1))

    image_points = np.array([(landmark[4], landmark[5]),  # nose
                             (landmark[10], landmark[11]),  # chin
                             (landmark[0], landmark[1]),  # l-eye
                             (landmark[2], landmark[3]),  # r-eye
                             (landmark[6], landmark[7]),  # l-mouse
                             (landmark[8], landmark[9])],  # r-mouse
                              dtype="float")

    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                  dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    rvec_matrix = cv2.Rodrigues(rotation_vector)[0]
    proj_matrix = np.hstack((rvec_matrix, translation_vector))

    euler_angles_radians = cv2.decomposeProjectionMatrix(proj_matrix)[6]

    pitch, yaw, roll = [theta for theta in euler_angles_radians]

    if pitch > 0:
        pitch = 180 - pitch
    elif pitch < 0:
        pitch = -180 - pitch

    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 500.0)]), rotation_vector, translation_vector,
                                                     camera_matrix, dist_coeffs)

    p1 = (int(image_points[0][0]), int(image_points[0][1]))
    p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

    return p1, p2, roll, yaw, pitch

def count_files(dir):
    joiner = (dir + os.path.sep).__add__

    return sum(
        os.path.isfile(filename)
        for filename
        in map(joiner, os.listdir(dir))
    )

def op_convolution(
    x, 
    w, b,
    fh, fw, 
    out_h, out_w, 
    stride_h, stride_w, 
    pads, 
    dtype="float32",
    transpose=False,
):
    n, _, _, _ = x.shape
    col = im2col(
        x, 
        fh, fw, 
        out_h, out_w, 
        stride_h, stride_w, 
        pads, 
        dtype=dtype,
    )
    col_w = transpose_kernel(w, transpose=transpose)

    out = np.dot(col, col_w)
    out = out.reshape(n, out_h, out_w, -1) + b
    return out

def op_pooling_max(
    x, 
    fh, fw, 
    out_h, out_w, 
    stride_h, stride_w, 
    pads, 
    dtype="float32",
):
    n, _, _, c = x.shape
    col = im2col(
        x, 
        fh, fw, 
        out_h, out_w, 
        stride_h, stride_w, 
        pads, 
        pad_value=np.min(x),
        dtype=dtype, 
    )
    col = col.reshape(-1, fh * fw)
    out = np.max(col, axis=1)
    out = out.reshape(n, out_h, out_w, c)
    return out

def op_fullyconnected(
    x, w, b,
    dtype="float32",
):
    x = x.reshape(-1, x.shape[-1])
    out = np.dot(x, w) + b
    return out

def op_mac(
    x, w, b,
    dtype="float32",
):
    out = w * x + b
    return out

def op_relu(
    x,
    alpha=0.0,
    dtype="float32",
):
    return np.where(x < 0, x * alpha, x)

def extend_inputs(
    x, 
    osize, 
    strides, 
    ksize, 
    dtype=np.float32
):
    b, h, w, c = x.shape
    oh, ow = osize
    sth, stw = strides
    ows = int((ow - w * stw) * 1.0 / 2.0)
    ohs = int((oh - h * sth) * 1.0 / 2.0)
    out = np.zeros((b, oh, ow, c), dtype=dtype)
    out[:, ohs::sth, ows::stw, :] = x
    return out

