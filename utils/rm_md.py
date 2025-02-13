


def remove_markdown_characters(text):

    # Replacing markdown characters with an empty string
    markdown_characters = ['*', '_', '~', '`', '>', '|', '!', '[', ']', '(', ')', '#', '-',"""\n""",'/',]
    for char in markdown_characters:
        text = text.replace(char, '')
    return text
