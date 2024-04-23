import constants
from models.handwriting_generation.word_set import WordSet
from PIL import Image


def generate_sentence(input_image_path, detected_objects, generated_text):
    word_set = WordSet('IIIT-HWS', constants.GENERATION_IMAGE_SIZE)
    input_image = Image.open(input_image_path).convert("RGB")
    input_image_width, input_image_height = input_image.size
    words = generated_text.split()
    last_word = detected_objects[-1]['words'][-1]
    starting_x = last_word['max_x'] + constants.INTRA_SENTENCE_PADDING
    starting_y = last_word['min_y']
    for word in words:
        image = word_set.get_image(word.lower(), image_size='in-line').convert("RGB")
        object_image_width, object_image_height = image.size
        if (starting_x + object_image_width + constants.INTRA_SENTENCE_PADDING) > input_image_width:
            starting_x = constants.LEFT_SENTENCE_PADDING
            starting_y = starting_y + constants.LINE_HEIGHT + constants.INTER_SENTENCE_PADDING
        input_image.paste(image, (starting_x, starting_y))
        starting_x = starting_x + object_image_width + constants.INTRA_SENTENCE_PADDING
    Image.save(input_image)
