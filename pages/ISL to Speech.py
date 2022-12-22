import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
from string import ascii_uppercase
import joblib
import mediapipe as mp
import numpy as np
import queue
from transformers import BertTokenizer,	BertForMaskedLM
from fast_autocomplete import AutoComplete
import torch
import string
from gtts import gTTS
from io import BytesIO


d = {'ONE': 0, 'TWO': 0, 'THREE': 0, 'FOUR': 0, 'FIVE': 0, 'SIX': 0, 'SEVEN': 0, 'EIGHT': 0,
     'NINE': 0, 'ZERO':	0, 'ACOMP':	0, 'BACKSPACE':	0, 'SPACE':	0, 'A':	0, 'B':	0, 'C':	0, 'D':	0,
     'E': 0, 'F': 0, 'G': 0, 'H': 0, 'I': 0, 'J': 0, 'K': 0, 'L': 0, 'M': 0, 'N': 0, 'O': 0, 'P': 0,
     'Q': 0, 'R': 0, 'S': 0, 'T': 0, 'U': 0, 'V': 0, 'W': 0, 'X': 0, 'Y': 0, 'Z': 0}
d1 = {'ONE': '1', 'TWO': '2', 'THREE': '3',	'FOUR':	'4', 'FIVE': '5', 'SIX': '6', 'SEVEN': '7',
      'EIGHT': '8',	'NINE':	'9', 'ZERO': '0'}

s = ""
temp = ""

for i in ascii_uppercase:
    d[i] = 0


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    min_detection_confidence=0.9, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

result_queue = (
    queue.Queue()
)


acomp_queue = (
    queue.Queue(maxsize=1)
)
# Prediction Part ******************************************************************


@st.cache()
def load_model(model_name):
    try:
        if model_name.lower() == "bert":
            bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            bert_model = BertForMaskedLM.from_pretrained(
                'bert-base-uncased').eval()
            return bert_tokenizer, bert_model
    except Exception as e:
        pass


def decode(tokenizer, pred_idx,	top_clean):
    ignore_tokens = string.punctuation + '[PAD]'
    tokens = []
    for w in pred_idx:
        token = ''.join(tokenizer.decode(w).split())
        if token not in ignore_tokens:
            tokens.append(token.replace('##', ''))
    return '\n'.join(tokens[:top_clean])


def encode(tokenizer, text_sentence, add_special_tokens=True):

    text_sentence = text_sentence.replace('<mask>', tokenizer.mask_token)
    if tokenizer.mask_token == text_sentence.split()[-1]:
        text_sentence += ' .'

        input_ids = torch.tensor(
            [tokenizer.encode(text_sentence, add_special_tokens=add_special_tokens)])
        mask_idx = torch.where(input_ids == tokenizer.mask_token_id)[
            1].tolist()[0]
    return input_ids, mask_idx


def get_all_predictions(text_sentence, top_clean=5):

    input_ids, mask_idx = encode(bert_tokenizer, text_sentence)
    with torch.no_grad():
        predict = bert_model(input_ids)[0]
    bert = decode(bert_tokenizer, predict[0, mask_idx, :].topk(
        top_k).indices.tolist(), top_clean)
    return {'bert':	bert}


def get_prediction_eos(input_text):

    try:
        input_text += '	<mask>'
        res = get_all_predictions(input_text, top_clean=int(top_k))
        return res
    except Exception as error:
        pass


def predict(temp):

    predictedword = temp.split(" ")[-1]

    words = {}

    with open(r"pages/wrds.txt") as f:
        for line in f:
            (key, val) = (line[:-1], {})
            words[key] = val

    autocomplete = AutoComplete(words=words)

    res = autocomplete.search(word=predictedword, max_cost=3, size=5)
    l = []

    for i in res:
        for j in i:
            l.append(j)
    return l[:5]


# OpenCV Part ************************************************************************************


def data_clean(landmark):

    data = landmark[0]
    try:
        data = str(data)
        data = data.strip().split('\n')
        garbage = ['landmark {', '  visibility:	0.0', '	 presence: 0.0', '}']
        without_garbage = []
        for i in data:
            if i not in garbage:
                without_garbage.append(i)
        clean = []
        for i in without_garbage:
            i = i.strip()
            clean.append(i[2:])
        for i in range(0, len(clean)):
            clean[i] = float(clean[i])
        return ([clean])
    except:
        return (np.zeros([1, 63], dtype=int)[0])


def process(image):
    global s
    global temp

    image = cv2.flip(image, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    frame = image
    image.flags.writeable = False
    results = hands.process(frame)
    image.flags.writeable = True

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        cleaned_landmark = data_clean(results.multi_hand_landmarks)

        if cleaned_landmark:

            clf = joblib.load(r"pages/finalized_model5.sav")

            y_pred = clf.predict(cleaned_landmark)
            image = cv2.putText(image, str(
                y_pred), (0, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 2, cv2.LINE_AA)
            if (y_pred[0] == "ACOMP"):
                d[y_pred[0]] += 1
                if d[y_pred[0]] > 10:
                    temp = y_pred[0]
                    d[y_pred[0]] = 0
            else:
                d[y_pred[0]] += 1
                if d[y_pred[0]] > 20:
                    if (y_pred[0] in d1.keys()):
                        if (s == ""):
                            s += d1[y_pred[0]]
                        else:
                            s += d1[y_pred[0]]
                        d[y_pred[0]] = 0
                    elif (y_pred[0] == 'BACKSPACE'):
                        s = s[:-1]
                        d[y_pred[0]] = 0
                    elif (y_pred[0] == 'SPACE'):
                        if ((s[-1] in ascii_uppercase) or (s[-1] in ('1234567890'))):
                            s += " "
                            d[y_pred[0]] = 0
                    else:
                        s += y_pred[0]
                        d[y_pred[0]] = 0
            result_queue.put(s.lower())
            acomp_queue.put(temp)

    return image


def callback(frame):
    img = frame.to_ndarray(format="bgr24")

    img = process(img)

    return av.VideoFrame.from_ndarray(img, format="bgr24")


def ttsgtts():
    sound_file = BytesIO()
    tts = gTTS(s, lang='en')
    tts.write_to_fp(sound_file)
    audwid.audio(sound_file)

webrtc_ctx = webrtc_streamer(key="example",	video_frame_callback=callback)


# Final	Main ******************************************************************************


if webrtc_ctx.state.playing:
    top_k = 3
    bert_tokenizer,	bert_model = load_model('BERT')

    with st.sidebar:

        st.markdown("Detection String -")
        placeholder = st.empty()
        st.markdown("Text Auto Complete	-")
        widget2 = st.empty()
        st.markdown("Next Word Prediction -")
        npwid = st.empty()

    result = ""
    st.checkbox("To	Speech", on_change=ttsgtts)
    st.markdown("Your Audio	Output -")
    audwid = st.empty()

    while True:
        try:

            placeholder.write(result)
            result = result_queue.get()
            temp2 = predict(result)
            queue2 = acomp_queue.get()
            splitres = result.split(" ")

            if (queue2 == "ACOMP" and result[-1].isdigit() == True and int(result[-1]) <= 4):

                if (splitres[-1].isdigit() == True):
                    autoges = nplist[int(result[-1])]

                else:
                    autoges = temp2[int(result[-1])]

                nowrds = result.split()
                with result_queue.mutex:
                    result_queue.queue.clear()
                with acomp_queue.mutex:
                    acomp_queue.queue.clear()

                if (len(nowrds) == 1):

                    s = autoges+" "
                    result_queue.put(s)

                else:

                    s = ""
                    for i in range(0, len(nowrds)-1):
                        s += nowrds[i]+" "
                    s += autoges+" "
                    result_queue.put(s)

            elif (splitres[-1] == ''):

                try:

                    if (len(splitres) > 1):
                        input_text = splitres[-2]
                        res = get_prediction_eos(input_text)
                        answer = []
                        nplist = res['bert'].split("\n")
                        for i in res['bert'].split("\n"):
                            answer.append(i)
                        answer_as_string = "	".join(answer)
                        npwid.write(nplist)

                except Exception as e:
                    print("SOME	PROBLEM	OCCURED")

            else:
                widget2.write(temp2)

        except queue.Empty:
            result = None
