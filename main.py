import cv2
import numpy as np

import constants
from CRAFT import object_detector
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image, ImageOps

if __name__ == '__main__':
    input_image_path = 'CRAFT/inputs/note.png'
    bboxes = object_detector.generate_bbox(input_image_path=input_image_path, cuda=False).astype(int)
    lines = object_detector.format_bbox_by_line(bboxes=bboxes)
    temp_words = [['terying', 'to', 'put', 'a', 'lot', 'letters', 'and', 'words'], ['in', 'this', 'sentence', 'and', 'test', 'different'], ['contexts']]

    input_image = Image.open(input_image_path).convert("RGB")
    # processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    # model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    for line_number, line in enumerate(lines):
        for word_number, word in enumerate(line['words']):
            min_bbox_dims = np.min(word['bbox'], axis=0)
            max_bbox_dims = np.max(word['bbox'], axis=0)
            object_image = input_image.crop((min_bbox_dims[0], min_bbox_dims[1], max_bbox_dims[0], max_bbox_dims[1]))
            right = 20
            left = 20
            top = 20
            bottom = 20

            width, height = object_image.size

            new_width = width + right + left
            new_height = height + top + bottom

            result = Image.new(object_image.mode, (new_width, new_height), (0, 0, 0))

            result.paste(object_image, (left, top))
            result = ImageOps.invert(result)

            # pixel_values = processor(result, return_tensors="pt").pixel_values
            # generated_ids = model.generate(pixel_values)
            # generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
            # recognized_text = generated_text[0]
            recognized_text = temp_words[line_number][word_number]

            lines[line_number]['words'][word_number]['recognized_text'] = recognized_text

            if height < constants.LINE_HEIGHT:
                padding_needed = constants.LINE_HEIGHT - height
                recognized_text_chars = set(recognized_text)
                padding_direction = 0 # -1=UP, 0=UP_AND_DOWN, 1=DOWN
                if len(recognized_text_chars.intersection(set(constants.UPPER_CHARACTERS))) > 0:
                    if len(recognized_text_chars.intersection(set(constants.LOWER_CHARACTERS))) == 0:
                        padding_direction = 1
                elif len(recognized_text_chars.intersection(set(constants.LOWER_CHARACTERS))) > 0:
                    if len(recognized_text_chars.intersection(set(constants.UPPER_CHARACTERS))) == 0:
                        padding_direction = -1
                top_padding = 0
                bottom_padding = 0
                if padding_direction == 0:
                    top_padding = padding_needed // 2
                    bottom_padding = padding_needed // 2
                if padding_direction == -1:
                    top_padding = padding_needed
                if padding_direction == 1:
                    bottom_padding = padding_needed
                lines[line_number]['words'][word_number]['bbox'][0, 1] -= top_padding
                lines[line_number]['words'][word_number]['bbox'][1, 1] -= top_padding
                lines[line_number]['words'][word_number]['bbox'][2, 1] += bottom_padding
                lines[line_number]['words'][word_number]['bbox'][3, 1] += bottom_padding
    object_detector.generate_image_with_bboxes(input_image_path, lines)


