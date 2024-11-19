import cv2
import mediapipe as mp
import os
import joblib
import numpy as np
from pythonosc import udp_client
import keyboard
import time
import warnings

# Global variables
midi_interval = 0.05  # Value of y to increase / decrease 1 midinote
respond_time = 1
mode_id = ["Frequency Mode", "Discrete Midi Mode"]
instrument = ["Sine oscillator", "Hand flute"]

# Initialize the OSC client
client = udp_client.SimpleUDPClient("127.0.0.1", 57120)


def sound_synth(midi_freq, mode, instrument_id):
    print(
        f"Sending Midinote Number: {midi_freq}, Mode: {mode}, Instrument: {instrument_id}"
    )
    client.send_message("/from_python", [midi_freq, mode, instrument_id])


script_dir = os.path.dirname(os.path.abspath(__file__))
model_file_path = os.path.join(script_dir, "gesture_model.pkl")
model = joblib.load(model_file_path)
gestures = ["fist", "open_hand"]
warnings.simplefilter("ignore")


def display_text(frame, text, position, color):
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def detect_hand_gesture(mode=0, midinote=81):
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    init = False
    key_pressed = False
    instrument_id = 0

    cap = cv2.VideoCapture(1)
    with mp_hands.Hands(
        max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7
    ) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture video. Exiting...")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            display_text(frame, f"Mode: {mode_id[mode]}", (10, 20), (255, 0, 0))
            display_text(
                frame,
                f"Instrument: {instrument[instrument_id]}",
                (10, 40),
                (255, 255, 0),
            )

            key = cv2.waitKey(respond_time) & 0xFF
            if key == ord("m"):
                mode = (mode + 1) % len(mode_id)
                init = False
            elif key == ord("i"):
                instrument_id = (instrument_id + 1) % len(instrument)
            elif key == ord("q"):
                print("Exiting the program...")
                break

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )
                    landmarks = (
                        np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark])
                        .flatten()
                        .reshape(1, -1)
                    )
                    gesture_id = model.predict(landmarks)[0]
                    gesture_text = gestures[int(gesture_id)]
                    middle_mcp = hand_landmarks.landmark[
                        mp_hands.HandLandmark.MIDDLE_FINGER_MCP
                    ]

                    display_text(
                        frame, f"Gesture: {gesture_text}", (10, 80), (0, 0, 255)
                    )

                    if mode == 0:
                        freq = np.clip((1 - middle_mcp.y) * 1980 + 20, 20, 2000)
                        display_text(
                            frame, f"Frequency: {freq:.0f}Hz", (10, 60), (0, 255, 0)
                        )
                        sound_synth(
                            0 if gesture_text == "fist" else freq, mode, instrument_id
                        )
                    elif mode == 1:
                        display_text(
                            frame,
                            f"Midi Note Number: {midinote}",
                            (10, 60),
                            (0, 255, 0),
                        )
                        if keyboard.is_pressed("s"):
                            mcp_y = middle_mcp.y
                            print("Position reset")
                            init = True
                        if init:
                            change_of_y = -(middle_mcp.y - mcp_y)
                            change_of_midi = int(change_of_y / midi_interval)
                            if abs(change_of_midi) > 0:
                                midinote += change_of_midi
                                mcp_y = middle_mcp.y
                        if gesture_text == "fist":
                            sound_synth(0, mode, instrument_id)
                        elif keyboard.is_pressed("z"):
                            if not key_pressed:
                                sound_synth(midinote, mode, instrument_id)
                                key_pressed = True
                        if not keyboard.is_pressed("z"):
                            key_pressed = False

            cv2.imshow("Hand Gesture Recognition", frame)

    cap.release()
    cv2.destroyAllWindows()


detect_hand_gesture(0, 81)
