import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import pyttsx3
import time
from collections import Counter

#LOAD MODEL
print("[INFO] Loading model...")
model = tf.keras.models.load_model("sign_model_tf")
_, MODEL_H, MODEL_W, _ = model.input_shape
print(f"[INFO] Model input size: {MODEL_H}x{MODEL_W}")


#CLASS NAMES
CLASS_NAMES = [
    'A','B','C','D','E','F','G','H','I','J','K','L','M',
    'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
    'del','nothing','space'
]


#MEDIAPIPE
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
detector = mp_hands.Hands(
    static_image_mode        = False,
    max_num_hands            = 1,
    min_detection_confidence = 0.75,
    min_tracking_confidence  = 0.65,
)

#TEXT-TO-SPEECH
engine = pyttsx3.init()
engine.setProperty('rate', 150)

#PREPROCESSING
def preprocess(roi_bgr, size):
    h, w = roi_bgr.shape[:2]
    if h == 0 or w == 0:
        return None, None

    canvas = np.ones((size, size, 3), dtype=np.uint8) * 255
    margin = 10
    scale  = (size - 2 * margin) / max(h, w)
    new_h  = max(1, int(h * scale))
    new_w  = max(1, int(w * scale))
    resized = cv2.resize(roi_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    y_off = (size - new_h) // 2
    x_off = (size - new_w) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized

    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = (rgb / 127.5) - 1.0
    return np.expand_dims(img, axis=0), canvas   # batch + preview


#TEMPORAL SMOOTHING
WINDOW         = 15
CONF_THRESHOLD = 0.80
pred_buffer    = []

def get_smoothed(label, conf):
    if conf < CONF_THRESHOLD:
        return None
    pred_buffer.append(label)
    if len(pred_buffer) > WINDOW:
        pred_buffer.pop(0)
    top, count = Counter(pred_buffer).most_common(1)[0]
    return top if (count / len(pred_buffer)) >= 0.60 else None

#SIGN LOCKING STATE MACHINE
SIGN_STATE   = "IDLE"    
locked_label = None      

def update_sentence(smoothed_label, sentence):
    """
    State machine: decides whether to add a letter to the sentence.
    Returns updated sentence and a status string for the UI.
    """
    global SIGN_STATE, locked_label

    if SIGN_STATE == "IDLE":
        if smoothed_label and smoothed_label != 'nothing':
            if smoothed_label == 'space':
                sentence.append(' ')
                print(f"[+] space  ->  {''.join(sentence)}")
            elif smoothed_label == 'del':
                if sentence:
                    removed = sentence.pop()
                    print(f"[-] deleted '{removed}'  ->  {''.join(sentence)}")
            else:
                sentence.append(smoothed_label)
                print(f"[+] '{smoothed_label}'  ->  {''.join(sentence)}")


            SIGN_STATE   = "LOCKED"
            locked_label = smoothed_label
            return sentence, f"ADDED: {smoothed_label}   (remove hand or change sign)"

        return sentence, None   
    
    elif SIGN_STATE == "LOCKED":
        if smoothed_label and smoothed_label != locked_label:
    
            SIGN_STATE   = "IDLE"
            locked_label = None
            pred_buffer.clear()   
            return sentence, "Ready for next sign..."


        return sentence, f"LOCKED: {locked_label}   (remove hand or change sign)"

    return sentence, None

def release_lock():
    global SIGN_STATE, locked_label
    if SIGN_STATE == "LOCKED":
        SIGN_STATE   = "IDLE"
        locked_label = None
        pred_buffer.clear()

#MAIN LOOP
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

sentence      = []
display_label = "Show a hand sign..."
status_msg    = ""

print("\n[READY]  Q=Quit  C=Backspace  X=Clear all  S=Speak  SPACE=Space")
print("[HOW TO USE] Hold sign -> wait for green ADDED bar -> remove hand -> next sign\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame     = cv2.flip(frame, 1)
    h_f, w_f  = frame.shape[:2]
    result    = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    hand_detected  = False
    preview_canvas = None

    if result.multi_hand_landmarks:
        for lms in result.multi_hand_landmarks:
            hand_detected = True
            mp_draw.draw_landmarks(frame, lms, mp_hands.HAND_CONNECTIONS)

            # Bounding box
            xs  = [int(l.x * w_f) for l in lms.landmark]
            ys  = [int(l.y * h_f) for l in lms.landmark]
            pad = 40
            x1  = max(0,   min(xs)-pad);  y1 = max(0,   min(ys)-pad)
            x2  = min(w_f, max(xs)+pad);  y2 = min(h_f, max(ys)+pad)
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,80), 2)

            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            img_batch, preview_canvas = preprocess(roi, MODEL_W)
            if img_batch is None:
                continue

            # Predict
            preds     = model.predict(img_batch, verbose=0)[0]
            idx       = int(np.argmax(preds))
            conf      = float(preds[idx])
            raw_label = CLASS_NAMES[idx]

            display_label = f"{raw_label} ({conf:.0%})"

            # Smooth
            smoothed = get_smoothed(raw_label, conf)

            # State machine — adds letter at most once per sign hold
            sentence, status = update_sentence(smoothed, sentence)
            if status:
                status_msg = status

    else:
        # No hand in frame
        release_lock()     
        pred_buffer.clear()
        display_label = "No hand"
        status_msg    = ""

   #UI
    sentence_str = "".join(sentence)

    # Top bar — current prediction
    cv2.rectangle(frame, (0,0), (w_f, 100), (20,20,20), -1)

    lbl_color = (0,255,100) if hand_detected else (120,120,120)
    cv2.putText(frame, f"Sign: {display_label}",
                (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, lbl_color, 2)

    # Status bar (LOCKED / ADDED / Detecting)
    if status_msg:
        bar_color = (0,180,60) if "ADDED" in status_msg else \
                    (0,100,200) if "LOCKED" in status_msg else \
                    (180,180,0)
        cv2.rectangle(frame, (0,50), (w_f, 78), bar_color, -1)
        cv2.putText(frame, status_msg,
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255,255,255), 1)

    cv2.putText(frame, f"Word: {sentence_str}",
                (12, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0,255,200), 2)

   
    if preview_canvas is not None:
        ph = pw = 90
        prev = cv2.resize(preview_canvas, (pw, ph))
        frame[h_f-32-ph : h_f-32, w_f-pw : w_f] = prev
        cv2.rectangle(frame, (w_f-pw, h_f-32-ph), (w_f, h_f-32), (0,255,80), 1)
        cv2.putText(frame, "Model sees",
                    (w_f-pw, h_f-32-ph-4),
                    cv2.FONT_HERSHEY_PLAIN, 0.85, (0,255,80), 1)

    # Bottom hint
    cv2.rectangle(frame, (0,h_f-30), (w_f,h_f), (20,20,20), -1)
    cv2.putText(frame, "Q=Quit  C=Backspace  X=Clear  S=Speak  SPACE=Space",
                (10,h_f-10), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (150,150,150), 1)

    cv2.imshow("ASL Real-time", frame)

    key = cv2.waitKey(1) & 0xFF
    if   key in (ord('q'), ord('Q')):
        break
    elif key in (ord('c'), ord('C')):
        if sentence:
            removed = sentence.pop()
            print(f"[INFO] Backspace — removed '{removed}'  ->  {''.join(sentence)}")
        pred_buffer.clear()
        SIGN_STATE   = "IDLE"
        locked_label = None
        status_msg   = ""
    elif key in (ord('x'), ord('X')):
       
        sentence.clear()
        pred_buffer.clear()
        SIGN_STATE   = "IDLE"
        locked_label = None
        status_msg   = ""
        print("[INFO] Sentence cleared.")
    elif key in (ord('s'), ord('S')):
        text = "".join(sentence).strip()
        if text:
            print(f"[SPEAK] '{text}'")
            import subprocess, sys
            script = (
                "import pyttsx3;"
                "e=pyttsx3.init();"
                "e.setProperty('rate',150);"
                f"e.say({repr(text)});"
                "e.runAndWait()"
            )
            subprocess.Popen([sys.executable, "-c", script])
        else:
            print("[SPEAK] Nothing to speak.")
    elif key == ord(' '):
        sentence.append(' ')
        print("[INFO] Space added.")

cap.release()
cv2.destroyAllWindows()
detector.close()
print("[INFO] Done.")