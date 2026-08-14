# 🤟 Real-Time Sign Language Recognition


## 📌 Project Overview


This project is a **real-time American Sign Language (ASL) recognition system** that uses deep learning and computer vision to recognize hand signs through a webcam.


The system uses **MobileNetV2 transfer learning** for image classification and **MediaPipe Hands** for real-time hand detection. Recognized signs are converted into letters and combined into words or sentences. The system also includes **text-to-speech functionality** to speak the generated sentence.


## 🎯 Features


- Real-time ASL sign recognition using a webcam
- Recognition of **26 alphabet signs (A-Z)**
- Hand detection and tracking using MediaPipe
- MobileNetV2-based deep learning classifier
- Two-phase training with transfer learning and fine-tuning
- Temporal smoothing to reduce unstable predictions
- Sign-locking mechanism to prevent duplicate letters
- Sentence formation using recognized signs
- Backspace and clear functionality
- Text-to-speech output
- Real-time prediction confidence display


## 🧠 Model Architecture


The project uses **MobileNetV2 pretrained on ImageNet** as the backbone.


The model first freezes the MobileNetV2 backbone and trains the classification layers. In the second phase, later layers of the backbone are unfrozen and fine-tuned using a lower learning rate.


### Classification Architecture


```text
MobileNetV2
     ↓
Global Average Pooling
     ↓
Dense (128, ReLU)
     ↓
Dropout (0.3)
     ↓
Dense (64, ReLU)
     ↓
Dropout (0.2)
     ↓
Dense (26, Softmax)
     ↓
ASL Letter Prediction


##🔤 Supported Signs

The model recognizes the 26 English alphabet signs:

A B C D E F G H I J K L M
N O P Q R S T U V W X Y Z

The real-time application also supports:

del
nothing
space

These additional labels are used for sentence construction and interaction.

##🛠️ Technologies Used
Python
TensorFlow
Keras
MobileNetV2
OpenCV
MediaPipe
NumPy
Matplotlib
Scikit-learn
pyttsx3


##📂 Project Structure
Sign-Language-Recognition/
│
├── .gitignore
├── LICENSE
├── README.md
│
├── best_asl_model.h5
├── best_phase1.keras
├── best_sign_model.keras
│
├── class_names.json
├── realtime_sign.py
├── sign_lang_demo.py
│
└── training_curves.png


##📊 Training Process

The model is trained in two phases.

Phase 1 — Transfer Learning

The MobileNetV2 backbone is frozen and the newly added classification layers are trained.

Configuration:

Image size: 224 × 224
Batch size: 64
Validation split: 15%
Data augmentation
Adam optimizer
Categorical cross-entropy loss

Data augmentation includes rotation, zoom, shifting, and brightness changes.

Phase 2 — Fine-Tuning

The later layers of the MobileNetV2 backbone are unfrozen and fine-tuned using a smaller learning rate.

The training process uses:

Early stopping
Learning-rate reduction
Model checkpointing
Validation accuracy monitoring


##📈 Training Results

Training and validation accuracy and loss are plotted and saved as:

training_curves.png

Training Curves

##🎥 Real-Time Recognition

The realtime_sign.py script uses MediaPipe Hands to detect a hand from the webcam and passes the detected hand region to the trained deep learning model.

The detected hand region is:

Detected using MediaPipe
Cropped using a bounding box
Resized and normalized
Passed to the trained model
Classified into an ASL sign
Smoothed across multiple predictions


##🔄 Prediction Stabilization

To reduce unstable frame-by-frame predictions, the system uses temporal smoothing.

Predictions are stored in a 15-frame window, and predictions below the confidence threshold are ignored.

The system also uses a sign-locking state machine so that holding the same sign does not repeatedly add the same letter to the sentence.

##🔊 Text-to-Speech

The application uses pyttsx3 to convert the generated sentence into speech.

Press:

S

to speak the current sentence.

##⌨️ Controls
Key	Function
Q	Quit application
C	Delete last character
X	Clear entire sentence
S	Speak sentence
SPACE	Add a space


##🚀 How to Run
1. Install Dependencies
pip install tensorflow opencv-python mediapipe numpy matplotlib scikit-learn pyttsx3
2. Train the Model

The training code is contained in:

python sign_lang_demo.py

The dataset is expected at:

dataset/asl_alphabet_train/

The dataset should be organized into folders corresponding to the ASL classes.

3. Run Real-Time Recognition

After the trained TensorFlow model has been generated:

python realtime_sign.py

Make sure your webcam is connected and accessible.

###🔍 System Workflow
              Webcam
                 ↓
       MediaPipe Hand Detection
                 ↓
         Hand Bounding Box
                 ↓
         Image Preprocessing
                 ↓
       MobileNetV2 Classifier
                 ↓
        ASL Letter Prediction
                 ↓
         Temporal Smoothing
                 ↓
           Sign Locking
                 ↓
        Sentence Formation
                 ↓
          Text-to-Speech
##💡 Applications
Sign language learning
Accessibility technology
Communication assistance
Educational tools
Human-computer interaction
Gesture-based interfaces


##🔮 Future Improvements
Support for dynamic ASL signs
Recognition of complete words and phrases
Multi-hand gesture recognition
Improved sentence prediction
Web or mobile deployment
Integration with a language model for sentence correction

##👩‍💻 Author

Nyasa Desai

Computer Science Student
SCET, Surat
