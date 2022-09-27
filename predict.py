import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from model import SSD300

from keras.applications.imagenet_utils import preprocess_input


def load_ssd_model(model_path, input_shape=[300, 300], num_classes=10):
    model_path = os.path.expanduser(model_path)
    assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
    
    ssd_model = SSD300([input_shape[0], input_shape[1], 3], num_classes)
    ssd_model.load_weights(model_path, by_name=True)
    print('{} model, anchors, and classes loaded.'.format(model_path))
    return ssd_model

def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 

#---------------------------------------------------#
#   ！！！以下代码请根据实际需要进行翻写修改！！！
#---------------------------------------------------#

def resize_image(image, size, letterbox_image):
    iw, ih = image.size
    w, h = size
    if letterbox_image:
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)

        image = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    else:
        new_image = image.resize((w, h), Image.BICUBIC)
    return new_image

class BBoxUtility(object):
    def __init__(self, num_classes, nms_thresh=0.45, top_k=300):
        self.num_classes = num_classes
        self._nms_thresh = nms_thresh
        self._top_k = top_k

    def nms(self, boxes, scores):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

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
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= self._nms_thresh)[0]
            order = order[inds + 1]

        return keep  

    def ssd_correct_boxes(self, box_xy, box_wh, input_shape, image_shape, letterbox_image):

        # 把y轴放前面是因为方便预测框和图像的宽高进行相乘
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        input_shape = np.array(input_shape)
        image_shape = np.array(image_shape)

        if letterbox_image:
            # 这里求出来的offset是图像有效区域相对于图像左上角的偏移情况
            # new_shape指的是宽高缩放情况
            new_shape = np.round(image_shape * np.min(input_shape/image_shape))
            offset = (input_shape - new_shape)/2./input_shape
            scale = input_shape/new_shape

            box_yx = (box_yx - offset) * scale
            box_hw *= scale

        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)
        boxes  = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]], axis=-1)
        boxes *= np.concatenate([image_shape, image_shape], axis=-1)
        return boxes

    def decode_boxes(self, mbox_loc, anchors, variances):
        # 获得先验框的宽与高
        anchor_width = anchors[:, 2] - anchors[:, 0]
        anchor_height = anchors[:, 3] - anchors[:, 1]
        # 获得先验框的中心点
        anchor_center_x = 0.5 * (anchors[:, 2] + anchors[:, 0])
        anchor_center_y = 0.5 * (anchors[:, 3] + anchors[:, 1])

        # 真实框距离先验框中心的xy轴偏移情况
        decode_bbox_center_x = mbox_loc[:, 0] * anchor_width * variances[0]
        decode_bbox_center_x += anchor_center_x
        decode_bbox_center_y = mbox_loc[:, 1] * anchor_height * variances[1]
        decode_bbox_center_y += anchor_center_y
        
        # 真实框的宽与高的求取
        decode_bbox_width  = np.exp(mbox_loc[:, 2] * variances[2])
        decode_bbox_width *= anchor_width
        decode_bbox_height = np.exp(mbox_loc[:, 3] * variances[3])
        decode_bbox_height *= anchor_height

        # 获取真实框的左上角与右下角
        decode_bbox_xmin = decode_bbox_center_x - 0.5 * decode_bbox_width
        decode_bbox_ymin = decode_bbox_center_y - 0.5 * decode_bbox_height
        decode_bbox_xmax = decode_bbox_center_x + 0.5 * decode_bbox_width
        decode_bbox_ymax = decode_bbox_center_y + 0.5 * decode_bbox_height

        # 真实框的左上角与右下角进行堆叠
        decode_bbox = np.concatenate((decode_bbox_xmin[:, None],
                                      decode_bbox_ymin[:, None],
                                      decode_bbox_xmax[:, None],
                                      decode_bbox_ymax[:, None]), axis=-1)
        # 防止超出0与1
        decode_bbox = np.minimum(np.maximum(decode_bbox, 0.0), 1.0)
        return decode_bbox

    def decode_box(self, predictions, anchors, image_shape, input_shape, letterbox_image, variances = [0.1, 0.1, 0.2, 0.2], confidence=0.5):
        # :4是回归预测结果
        mbox_loc = predictions[:, :, :4]
        # 获得种类的置信度
        mbox_conf = predictions[:, :, 4:]

        results = []
        # 对每一张图片进行处理，由于在predict.py的时候，我们只输入一张图片，所以for i in range(len(mbox_loc))只进行一次
        for i in range(len(mbox_loc)):
            results.append([])
            # 利用回归结果对先验框进行解码
            decode_bbox = self.decode_boxes(mbox_loc[i], anchors, variances)

            for c in range(1, self.num_classes):
                # 取出属于该类的所有框的置信度
                # 判断是否大于门限
                c_confs = mbox_conf[i, :, c]
                c_confs_m = c_confs > confidence
                if len(c_confs[c_confs_m]) > 0:
                    # 取出得分高于confidence的框
                    boxes_to_process = decode_bbox[c_confs_m]
                    confs_to_process = c_confs[c_confs_m]

                    # 进行iou的非极大抑制
                    idx = self.nms(boxes_to_process, confs_to_process, )

                    # 取出在非极大抑制中效果较好的内容
                    good_boxes = boxes_to_process[idx]
                    confs = confs_to_process[idx][:, None]
                    labels = (c - 1) * np.ones((len(idx), 1))

                    # 将label、置信度、框的位置进行堆叠。
                    c_pred = np.concatenate((good_boxes, labels, confs), axis=1)
                    # 添加进result里
                    results[-1].extend(c_pred)

            if len(results[-1]) > 0:
                results[-1] = np.array(results[-1])
                box_xy, box_wh = (results[-1][:, 0:2] + results[-1][:, 2:4])/2, results[-1][:, 2:4] - results[-1][:, 0:2]
                results[-1][:, :4] = self.ssd_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)

        return results


class AnchorBox():
    def __init__(self, input_shape, min_size, max_size=None, aspect_ratios=None, flip=True):
        self.input_shape = input_shape

        self.min_size = min_size
        self.max_size = max_size

        self.aspect_ratios = []
        for ar in aspect_ratios:
            self.aspect_ratios.append(ar)
            self.aspect_ratios.append(1.0 / ar)

    def call(self, layer_shape, mask=None):

        # 获取输入进来的特征层的宽和高
        # 比如38x38
        layer_height = layer_shape[0]
        layer_width = layer_shape[1]

        # 获取输入进来的图片的宽和高
        # 比如300x300
        img_height = self.input_shape[0]
        img_width = self.input_shape[1]

        box_widths = []
        box_heights = []

        # self.aspect_ratios一般有两个值
        # [1, 1, 2, 1/2]
        # [1, 1, 2, 1/2, 3, 1/3]
        for ar in self.aspect_ratios:
            # 首先添加一个较小的正方形
            if ar == 1 and len(box_widths) == 0:
                box_widths.append(self.min_size)
                box_heights.append(self.min_size)
            # 然后添加一个较大的正方形
            elif ar == 1 and len(box_widths) > 0:
                box_widths.append(np.sqrt(self.min_size * self.max_size))
                box_heights.append(np.sqrt(self.min_size * self.max_size))
            # 然后添加长方形
            elif ar != 1:
                box_widths.append(self.min_size * np.sqrt(ar))
                box_heights.append(self.min_size / np.sqrt(ar))

        # 获得所有先验框的宽高1/2
        box_widths = 0.5 * np.array(box_widths)
        box_heights = 0.5 * np.array(box_heights)

        # 每一个特征层对应的步长
        step_x = img_width / layer_width
        step_y = img_height / layer_height

        # 生成网格中心
        linx = np.linspace(0.5 * step_x, img_width - 0.5 * step_x,
                           layer_width)
        liny = np.linspace(0.5 * step_y, img_height - 0.5 * step_y,
                           layer_height)
        centers_x, centers_y = np.meshgrid(linx, liny)
        centers_x = centers_x.reshape(-1, 1)
        centers_y = centers_y.reshape(-1, 1)

        # 每一个先验框需要两个(centers_x, centers_y)，前一个用来计算左上角，后一个计算右下角
        num_anchors_ = len(self.aspect_ratios)
        anchor_boxes = np.concatenate((centers_x, centers_y), axis=1)
        anchor_boxes = np.tile(anchor_boxes, (1, 2 * num_anchors_))
        
        # 获得先验框的左上角和右下角
        anchor_boxes[:, ::4]    -= box_widths
        anchor_boxes[:, 1::4]   -= box_heights
        anchor_boxes[:, 2::4]   += box_widths
        anchor_boxes[:, 3::4]   += box_heights

        # 将先验框变成小数的形式
        # 归一化
        anchor_boxes[:, ::2]    /= img_width
        anchor_boxes[:, 1::2]   /= img_height
        anchor_boxes = anchor_boxes.reshape(-1, 4)

        anchor_boxes = np.minimum(np.maximum(anchor_boxes, 0.0), 1.0)
        return anchor_boxes

def get_img_output_length(height, width):
    filter_sizes    = [3, 3, 3, 3, 3, 3, 3, 3]
    padding         = [1, 1, 1, 1, 1, 1, 0, 0]
    stride          = [2, 2, 2, 2, 2, 2, 1, 1]
    feature_heights = []
    feature_widths  = []

    for i in range(len(filter_sizes)):
        height  = (height + 2*padding[i] - filter_sizes[i]) // stride[i] + 1
        width   = (width + 2*padding[i] - filter_sizes[i]) // stride[i] + 1
        feature_heights.append(height)
        feature_widths.append(width)
    return np.array(feature_heights)[-6:], np.array(feature_widths)[-6:]

def get_anchors(input_shape = [300, 300], anchors_size = [30, 60, 111, 162, 213, 264, 315]):
    feature_heights, feature_widths = get_img_output_length(input_shape[0], input_shape[1])
    aspect_ratios = [[1, 2], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2], [1, 2]]
    anchors = []
    for i in range(len(feature_heights)):
        anchors.append(AnchorBox(input_shape, anchors_size[i], max_size = anchors_size[i+1], 
                    aspect_ratios = aspect_ratios[i]).call([feature_heights[i], feature_widths[i]]))

    anchors = np.concatenate(anchors, axis=0)
    return anchors


def predict(image, model, bbox_util, anchors, input_shape=[300, 300], confidence=0.5):
    # 计算输入图片的高和宽
    image_shape = np.array(np.shape(image)[0:2])

    # 在这里将图像转换成RGB图像，防止灰度图在预测时报错。
    # 代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
    image = cvtColor(image)

    # 给图像增加灰条，实现不失真的resize
    # 也可以直接resize进行识别
    # letterbox_image参数用于控制是否使用letterbox_image对输入图像进行不失真的resize，在多次测试后，发现关闭letterbox_image直接resize的效果更好
    image_data = resize_image(image, (input_shape[1], input_shape[0]), letterbox_image=False)

    # 添加上batch_size维度，图片预处理，归一化。
    image_data = preprocess_input(np.expand_dims(np.array(image_data, dtype='float32'), 0))

    # 模型计算推理
    preds = model.predict(image_data)
    # 将预测结果进行解码
    results = bbox_util.decode_box(preds, anchors, image_shape, input_shape, letterbox_image=False, confidence=confidence)

    return results

if __name__ == '__main__':
    # 加载类别
    class_names = []

    input_shape = [300, 300]

    # 加载模型
    model = load_ssd_model('model.h5', input_shape=input_shape, num_classes=len(class_names))

    # 加载先验框
    anchors = get_anchors(input_shape=input_shape, anchors_size=[21, 45, 99, 153, 207, 261, 315])

    # 加载bbox_util
    bbox_util = BBoxUtility(len(class_names), nms_thresh=0.45)

    # 读取图片
    image = Image.open('test.jpg')

    # 预测
    results = predict(image, model, bbox_util, anchors, confidence=0.5)
    top_label = results[0][:, 4]
    top_conf = results[0][:, 5]
    top_boxes = results[0][:, :4]

    # 输出结果
    for i, c in list(enumerate(top_label)):
            predicted_class = class_names[int(c)]
            box = top_boxes[i]
            score = str(top_conf[i])
            
            top, left, bottom, right = box

            if predicted_class not in class_names:
                continue

            print("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

    # 画框
    import colorsys
    hsv_tuples = [(x / len(class_names), 1., 1.) for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    font = ImageFont.truetype(font='simhei.ttf', size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))
    thickness = max((np.shape(image)[0] + np.shape(image)[1]) // input_shape[0], 1)
    for i, c in list(enumerate(top_label)):
        predicted_class = class_names[int(c)]
        box = top_boxes[i]
        score = top_conf[i]

        top, left, bottom, right = box

        top     = max(0, np.floor(top).astype('int32'))
        left    = max(0, np.floor(left).astype('int32'))
        bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
        right   = min(image.size[0], np.floor(right).astype('int32'))

        label = '{} {:.2f}'.format(predicted_class, score)
        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)
        label = label.encode('utf-8')
        print(label, top, left, bottom, right)
        
        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        for i in range(thickness):
            draw.rectangle([left + i, top + i, right - i, bottom - i], outline=colors[c])
        draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=colors[c])
        draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
        del draw

    image.save('result.jpg', quality=95, subsampling=0)


    




