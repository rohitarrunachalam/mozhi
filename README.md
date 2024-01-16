# Mozhi 
An AI Assistance Tool for Sign Language Conversion !

### Table of Contents  
- [Overview](#Overview)  
- [Working](#Working) 
- [Demo](#Demo) 
- [Installation](#Installation) 
- [Run](#Run) 
- [Technologies Used](#Technologies%Used) 
- [Got a Question](#Got%a%Question%?) 



### Overview
A Web Application based Sign Language to Speech and vice-versa conversion tool that helps in bridging the communication gap between deaf-mute and normal
people.


### Working

- The User's hand movements are captured with the help of OpenCV and we use Mediapipe to extract the coordinates.
- These coordinated are then compared with the trained SVM Model(which we trained with our own datasets) and the appropriate letter is displayed as output in the screen.
- After each detection the letter gets appended to the string forming a word. Parallely we also included features like Auto-Complete and Next Word Prediction using BERT to make the process more efficient and faster.
- Finally, the output word is converted to speech and displayed as output.

### Demo
![](https://github.com/harikrish-s/Sign-Language-Recognition/blob/main/demo/demo-pic.png)

### Installation

It is recommended to use a Virtual Environment to run this project

Install all the required packages using
```
pip install -r requirements.txt
```
### To run the code

Open terminal in the directory where the home.py file is present and run the command
```
streamlit run app.py
```

### Technologies Used


![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white) ![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)



### Got a Question ?

Feel free to contact us on LinkedIn - [Harikrishnan S](https://www.linkedin.com/in/harikrishnan-s-580461214/) , [Rohit Arrunachalam](https://www.linkedin.com/in/rohitarrunachalam/) , [Isha](https://www.linkedin.com/in/isha-reddy-vaka-1457a9228/).

Make a pull request on this repo if you would like work towards improving this project.

