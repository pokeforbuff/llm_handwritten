import cv2
import numpy as np

from CRAFT import object_detector
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image, ImageOps

if __name__ == '__main__':
    input_image_path = 'CRAFT/inputs/note.png'
    input_image = Image.open(input_image_path).convert("RGB")
    bboxes = object_detector.generate_bbox(input_image_path=input_image_path, cuda=False).astype(int)
    lines = object_detector.format_bbox_by_line(input_image_path=input_image_path, bboxes=bboxes)
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    for line in lines:
        for word in line['words']:
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

            pixel_values = processor(result, return_tensors="pt").pixel_values
            generated_ids = model.generate(pixel_values)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
            print(generated_text[0])
