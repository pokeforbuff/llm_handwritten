import constants
from models.handwriting_generation.word_set import WordSet
from PIL import Image
import os


def generate_sentence(input_image_path, detected_objects, generated_text):
    word_set = WordSet('IIIT-HWS', constants.GENERATION_IMAGE_SIZE)
    input_image = Image.open(input_image_path).convert("RGBA")
    input_image_width, input_image_height = input_image.size
    words = generated_text.split()
    last_word = detected_objects[-1]['words'][-1]
    last_word_height = int(last_word['max_y'] - last_word['min_y'])
    starting_x = last_word['max_x'] + constants.INTRA_SENTENCE_PADDING
    starting_y = last_word['min_y']
    for word in words:
        image = word_set.get_image(word.lower(), image_size='in-line', line_height=last_word_height)
        object_image_width, object_image_height = image.size
        if (starting_x + object_image_width + constants.INTRA_SENTENCE_PADDING) > input_image_width:
            starting_x = constants.LEFT_SENTENCE_PADDING
            starting_y = starting_y + last_word_height + constants.INTER_SENTENCE_PADDING
        input_image.paste(image, (int(starting_x), int(starting_y)), mask=image)
        starting_x = starting_x + object_image_width + constants.INTRA_SENTENCE_PADDING
    filename, file_ext = os.path.splitext(os.path.basename(input_image_path))
    input_image.save('result/final_res_' + filename + '.png')
