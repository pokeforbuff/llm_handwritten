from transformers import pipeline


def generate(
        text,
        num_words_to_generate=1
):
    generated_text = ''
    unmasker = pipeline('fill-mask', model='bert-large-cased')
    for i in range(num_words_to_generate):
        generated_words = unmasker(f'{text} {generated_text}[MASK].')
        best_word = generated_words[0]['token_str']
        generated_text += f'{best_word} '
    return generated_text
