import re

def clean_text_light(text):
    #Conversión a minúsculas
    text = text.lower()
    #Sustituir URLs por tokens genéricos
    text = re.sub(r'http\S+|www\S+|wap\S+', 'URL', text)  # reemplaza URLs
    #text = re.sub(r'\d+', 'NUM', text)      # reemplaza números
    #Limpia espacios
    text = re.sub(r'\s+', ' ', text).strip()
    #no eliminamos signos, emojis ni abreviaturas
    return text