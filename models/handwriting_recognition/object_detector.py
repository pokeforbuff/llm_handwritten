"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import os
import time
import argparse

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import cv2
import numpy as np
from models.handwriting_recognition import imgproc, file_utils, craft_utils
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image, ImageOps
import constants

from models.handwriting_recognition.craft import CRAFT

from collections import OrderedDict


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")


parser = argparse.ArgumentParser(description='handwriting_recognition Text Detection')
parser.add_argument('--trained_model', default='weights/craft_mlt_25k.pth', type=str, help='pretrained model')
parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda for inference')
parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
parser.add_argument('--test_folder', default='/data/', type=str, help='folder path to input images')
parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str,
                    help='pretrained refiner model')

args = parser.parse_args()


def test_net(net, image, canvas_size, text_threshold, link_threshold, low_text, cuda, poly, show_time, mag_ratio,
             refine_net=None):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, canvas_size,
                                                                          interpolation=cv2.INTER_LINEAR,
                                                                          mag_ratio=mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))  # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link = y[0, :, :, 1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0, :, :, 0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    if show_time: print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text


def generate_bbox(
        input_image_path=None,
        result_folder='result/',
        trained_model='models/handwriting_recognition/weights/craft_mlt_25k.pth',
        text_threshold=0.7,
        low_text=0.4,
        link_threshold=0.4,
        cuda=True,
        canvas_size=1280,
        mag_ratio=1.5,
        poly=False,
        show_time=False,
        refine=False,
        refiner_model='models/handwriting_recognition/weights/craft_refiner_CTW1500.pth'
):

    if not os.path.isdir(result_folder):
        os.mkdir(result_folder)
    net = CRAFT()  # initialize
    if cuda:
        net.load_state_dict(copyStateDict(torch.load(trained_model)))
    else:
        net.load_state_dict(copyStateDict(torch.load(trained_model, map_location='cpu')))

    if cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()

    # LinkRefiner
    refine_net = None
    if refine:
        from refinenet import RefineNet

        refine_net = RefineNet()
        if cuda:
            refine_net.load_state_dict(copyStateDict(torch.load(refiner_model)))
            refine_net = refine_net.cuda()
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(copyStateDict(torch.load(refiner_model, map_location='cpu')))

        refine_net.eval()
        poly = True
    image = imgproc.loadImage(input_image_path)

    bboxes, polys, score_text = test_net(net, image, canvas_size, text_threshold, link_threshold, low_text, cuda,
                                         poly,
                                         show_time,
                                         mag_ratio,
                                         refine_net)
    filename, file_ext = os.path.splitext(os.path.basename(input_image_path))
    mask_file = result_folder + "/res_" + filename + '_mask.jpg'
    cv2.imwrite(mask_file, score_text)

    file_utils.saveResult(input_image_path, image[:, :, ::-1], polys, dirname=result_folder)

    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    input_image = Image.open(input_image_path).convert("RGB")
    img = cv2.imread(input_image_path)
    num_bboxes = bboxes.shape[0]
    words = []
    for i in range(num_bboxes):
        word = {}
        min_x_y = bboxes[i].min(axis=0)
        max_x_y = bboxes[i].max(axis=0)
        min_x = min_x_y[0]
        min_y = min_x_y[1]
        max_x = max_x_y[0]
        max_y = max_x_y[1]
        word['word_number'] = None
        word['min_x'] = min_x
        word['min_y'] = min_y
        word['max_x'] = max_x
        word['max_y'] = max_y
        words.append(word)
    lines = {
        0: {
            'line_number': 0,
            'min_y': 0,
            'max_y': 0,
            'words': []
        }
    }
    recognized_final_text = ''
    for i, word in enumerate(words):
        line_assignment = 0
        for line_number in range(1, len(lines) + 1):
            if word['min_y'] >= lines[line_number - 1]['max_y']:
                line_assignment = line_number
            else:
                break
        if line_assignment not in lines:
            lines[line_assignment] = {
                'line_number': line_assignment,
                'recognized_text': '',
                'min_y': 0,
                'max_y': 0,
                'words': []
            }
        lines[line_assignment]['words'].append(word)
        if lines[line_assignment]['min_y'] > 0:
            lines[line_assignment]['min_y'] = min(lines[line_assignment]['min_y'], word['min_y'])
        else:
            lines[line_assignment]['min_y'] = word['min_y']
        if lines[line_assignment]['max_y'] > 0:
            lines[line_assignment]['max_y'] = max(lines[line_assignment]['max_y'], word['max_y'])
        else:
            lines[line_assignment]['max_y'] = word['max_y']
    del lines[0]
    for line_number in range(1, len(lines) + 1):
        top_padding = 0
        bottom_padding = 0
        lines[line_number]['words'] = sorted(lines[line_number]['words'], key=lambda x: x['min_x'])
        for i, word in enumerate(lines[line_number]['words']):
            lines[line_number]['words'][i]['word_number'] = i + 1
            lines[line_number]['words'][i]['min_y'] = lines[line_number]['min_y']
            lines[line_number]['words'][i]['max_y'] = lines[line_number]['max_y']
            lines[line_number]['words'][i]['bbox'] = np.array([
                [word['min_x'], word['min_y']],
                [word['max_x'], word['min_y']],
                [word['max_x'], word['max_y']],
                [word['min_x'], word['max_y']]
            ])
            min_bbox_dims = np.min(lines[line_number]['words'][i]['bbox'], axis=0)
            max_bbox_dims = np.max(lines[line_number]['words'][i]['bbox'], axis=0)
            object_image = input_image.crop((min_bbox_dims[0], min_bbox_dims[1], max_bbox_dims[0], max_bbox_dims[1]))
            right = 20
            left = 20
            top = 20
            bottom = 20
            width, height = object_image.size
            new_width = width + right + left
            new_height = height + top + bottom
            result = Image.new(object_image.mode, (new_width, new_height), "white" if constants.THEME == 'light' else "black")
            result.paste(object_image, (left, top))
            if constants.THEME == 'dark':
                result = ImageOps.invert(result)
            pixel_values = processor(result, return_tensors="pt").pixel_values
            generated_ids = model.generate(pixel_values)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
            recognized_text = generated_text[0]
            if recognized_text.endswith('.'):
                recognized_text = recognized_text[:-1]
            lines[line_number]['words'][i]['recognized_text'] = recognized_text
            lines[line_number]['recognized_text'] += f'{recognized_text} '
            recognized_final_text += f'{recognized_text} '
        recognized_text_chars = set(lines[line_number]['recognized_text'])
        if len(recognized_text_chars.intersection(set(constants.UPPER_CHARACTERS))) > 0:
            if len(recognized_text_chars.intersection(set(constants.LOWER_CHARACTERS))) == 0:
                lines[line_number]['max_y'] += (lines[line_number]['max_y'] - lines[line_number]['min_y']) // 4
        elif len(recognized_text_chars.intersection(set(constants.LOWER_CHARACTERS))) > 0:
            if len(recognized_text_chars.intersection(set(constants.UPPER_CHARACTERS))) == 0:
                lines[line_number]['min_y'] -= (lines[line_number]['max_y'] - lines[line_number]['min_y']) // 4
        if lines[line_number]['recognized_text'] != '':
            lines[line_number]['recognized_text'] = lines[line_number]['recognized_text'][:-1]
        words.append(lines[line_number]['words'])
        for i, word in enumerate(lines[line_number]['words']):
            lines[line_number]['words'][i]['min_y'] = lines[line_number]['min_y']
            lines[line_number]['words'][i]['max_y'] = lines[line_number]['max_y']
            lines[line_number]['words'][i]['bbox'] = np.array([
                [word['min_x'], word['min_y']],
                [word['max_x'], word['min_y']],
                [word['max_x'], word['max_y']],
                [word['min_x'], word['max_y']]
            ])
            poly = np.array(lines[line_number]['words'][i]['bbox']).astype(np.int32).reshape((-1))
            poly = poly.reshape(-1, 2)
            cv2.polylines(img, [poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)
    bboxes = list(lines.values())
    result_file = result_folder + "/formatted_res_" + filename + '.jpg'
    cv2.imwrite(result_file, img)

    return bboxes, recognized_final_text
