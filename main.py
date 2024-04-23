from models import handwriting_recognition, text_autocompletion, handwriting_generation

if __name__ == '__main__':
    input_image_path = 'inputs/note.png'
    detected_objects, recognized_text = handwriting_recognition.run(input_image_path=input_image_path, cuda=True)
    generated_text = text_autocompletion.run(recognized_text)
    handwriting_generation.run(input_image_path, detected_objects, generated_text)
