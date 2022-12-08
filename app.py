import cv2
import numpy as np
import av
import time
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import joblib
import streamlit as st
from string import ascii_uppercase



d = {'ONE': 0, 'TWO': 0, 'THREE': 0, 'FOUR': 0, 'FIVE': 0, 'SIX': 0, 'SEVEN': 0, 'EIGHT': 0,
	'NINE': 0, 'ZERO': 0, 'ACOMP': 0, 'BACKSPACE': 0, 'SPACE': 0, 'A': 0, 'B': 0, 'C': 0, 'D': 0,
	'E': 0, 'F': 0, 'G': 0, 'H': 0, 'I': 0, 'J': 0, 'K': 0, 'L': 0, 'M': 0, 'N': 0, 'O': 0, 'P': 0,
	'Q': 0, 'R': 0, 'S': 0, 'T': 0, 'U': 0, 'V': 0, 'W': 0, 'X': 0, 'Y': 0, 'Z': 0}
d1 = {'ONE': '1', 'TWO': '2', 'THREE': '3', 'FOUR': '4', 'FIVE': '5', 'SIX': '6', 'SEVEN': '7',
	'EIGHT': '8', 'NINE': '9', 'ZERO': '0'}

s = ""
for i in ascii_uppercase:
	d[i]=0


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    min_detection_confidence=0.9, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)


def data_clean(landmark):
  
  data = landmark[0]
  try:
    data = str(data)
    data = data.strip().split('\n')
    garbage = ['landmark {', '  visibility: 0.0', '  presence: 0.0', '}']
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
    return([clean])
  except:
    return(np.zeros([1,63], dtype=int)[0])


def process(image): 
    global s
    
    image = cv2.flip(image, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    frame=image
    image.flags.writeable = False
    results = hands.process(frame)
    image.flags.writeable = True
   
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.rectangle(image,(0,0),(400,100),(255, 255, 255),-1)
    cv2.rectangle(image,(0,428),(704,528),(255, 255, 255),-1)
    if results.multi_hand_landmarks:
        cleaned_landmark = data_clean(results.multi_hand_landmarks)
       
        if cleaned_landmark:
            
            clf = joblib.load("finalized_model5.sav")
           
            y_pred = clf.predict(cleaned_landmark)
            classes_x=np.argmax(y_pred)
            image = cv2.putText(image, str(y_pred), (0,70),cv2.FONT_HERSHEY_SIMPLEX,  3, (0,0,0), 2, cv2.LINE_AA)
            d[y_pred[0]]+=1
            widget = st.empty()
            if d[y_pred[0]]>20:
                if (y_pred[0] in d1.keys()):
                    if (s==""):
                        s+=d1[y_pred[0]]
                    elif (s[-1] in ascii_uppercase):
                        s+=" "+d1[y_pred[0]]
                    else:
                        s+=d1[y_pred[0]]
                    d[y_pred[0]]=0
                elif (y_pred[0]=='BACKSPACE'):
                    s=s[:-1]
                    d[y_pred[0]]=0
                elif (y_pred[0]=='SPACE'):
                    if ((s[-1] in ascii_uppercase ) or (s[-1] in ('1234567890'))):
                        s+=" "
                        d[y_pred[0]]=0
                else:
                    s+=y_pred[0]
                    d[y_pred[0]]=0
            print("**************************************************\n")
            print(s)
            print("**************************************************\n")
            widget.write(s)
   
    image = cv2.putText(image, s, (0, 465), cv2.FONT_HERSHEY_SIMPLEX,1, 
                            (0, 0, 0),2, cv2.LINE_AA, False)                             

    return image


RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")    
        img = process(img)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_ctx = webrtc_streamer(
    key="WYH",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=VideoProcessor,
    async_processing=True,
)



