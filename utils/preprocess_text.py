import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils.clean_text_light import clean_text_light

def preprocess_text(messages, tokenizer, max_len=40):
    
    #Limpia, tokeniza y paddea los mensajes para el modelo.
    msg_clean = [clean_text_light(msg) for msg in messages]
    sequences = tokenizer.texts_to_sequences(msg_clean)
    padded = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    return padded
