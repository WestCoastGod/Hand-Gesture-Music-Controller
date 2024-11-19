import cv2
import mediapipe as mp
import os
import joblib
import numpy as np
from pythonosc import udp_client
import keyboard
import warnings
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# Global variables
midi_interval = 0.05  # Value of y to increase / decrease 1 midinote
respond_time = 1
mode_id = ["Frequency Mode", "Discrete Midi Mode"]
instrument = ["Sine oscillator", "Hand flute"]

# Initialize the OSC client
client = udp_client.SimpleUDPClient("127.0.0.1", 57120)

script_dir = os.path.dirname(os.path.abspath(__file__))
model_file_path = os.path.join(script_dir, "gesture_model.pkl")
model = joblib.load(model_file_path)
gestures = ["fist", "open_hand"]
warnings.simplefilter("ignore")


class HandGestureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hand Gesture Music Controller")

        self.mode = 0
        self.midinote = 81
        self.instrument_id = 0
        self.init = False
        self.key_pressed = False
        self.mcp_y = 0

        self.title_label = ttk.Label(
            root, text="Hand Gesture Recognition", font=("Helvetica", 16)
        )
        self.title_label.pack(pady=10)

        self.video_frame = ttk.Frame(root)
        self.video_frame.pack(side=tk.LEFT, padx=10)

        self.info_frame = ttk.Frame(root)
        self.info_frame.pack(side=tk.RIGHT, padx=10)

        self.mode_label = ttk.Label(self.info_frame, text=f"Mode: {mode_id[self.mode]}")
        self.mode_label.pack(pady=5)

        self.instrument_label = ttk.Label(
            self.info_frame, text=f"Instrument: {instrument[self.instrument_id]}"
        )
        self.instrument_label.pack(pady=5)

        self.frequency_label = ttk.Label(self.info_frame, text="Frequency: N/A")
        self.frequency_label.pack(pady=5)

        self.gesture_label = ttk.Label(self.info_frame, text="Gesture: N/A")
        self.gesture_label.pack(pady=5)

        self.mode_button = ttk.Button(
            self.info_frame, text="Change Mode", command=self.change_mode
        )
        self.mode_button.pack(pady=5)

        self.instrument_button = ttk.Button(
            self.info_frame, text="Change Instrument", command=self.change_instrument
        )
        self.instrument_button.pack(pady=5)

        self.reset_button = ttk.Button(
            self.info_frame, text="Reset Position", command=self.reset_position
        )
        self.reset_button.pack(pady=5)

        self.video_label = ttk.Label(self.video_frame)
        self.video_label.pack()

        self.cap = cv2.VideoCapture(1)
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

        self.update_video()

    def change_mode(self):
        self.mode = (self.mode + 1) % len(mode_id)
        self.mode_label.config(text=f"Mode: {mode_id[self.mode]}")
        self.init = False

    def change_instrument(self):
        self.instrument_id = (self.instrument_id + 1) % len(instrument)
        self.instrument_label.config(
            text=f"Instrument: {instrument[self.instrument_id]}"
        )

    def reset_position(self):
        self.init = True
        print("Position reset")

    def update_mcp_y(self, mcp_y):
        self.mcp_y = mcp_y

    def update_video(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to capture video. Exiting...")
            self.root.quit()
            return

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        with self.mp_hands.Hands(
            max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7
        ) as hands:
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )
                    landmarks = (
                        np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark])
                        .flatten()
                        .reshape(1, -1)
                    )
                    gesture_id = model.predict(landmarks)[0]
                    gesture_text = gestures[int(gesture_id)]
                    middle_mcp = hand_landmarks.landmark[
                        self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP
                    ]

                    self.gesture_label.config(text=f"Gesture: {gesture_text}")

                    if self.mode == 0:
                        freq = (
                            0
                            if gesture_text == "fist"
                            else np.clip((1 - middle_mcp.y) * 1980 + 20, 20, 2000)
                        )
                        self.frequency_label.config(text=f"Frequency: {freq:.0f}Hz")
                        sound_synth(
                            0 if gesture_text == "fist" else freq,
                            self.mode,
                            self.instrument_id,
                        )
                    elif self.mode == 1:
                        self.frequency_label.config(
                            text=f"Midi Note Number: {self.midinote}"
                        )
                        if self.init:
                            self.update_mcp_y(middle_mcp.y)
                            self.init = False
                        change_of_y = -(middle_mcp.y - self.mcp_y)
                        change_of_midi = int(change_of_y / midi_interval)
                        if abs(change_of_midi) > 0:
                            self.midinote += change_of_midi
                            self.update_mcp_y(middle_mcp.y)
                        if gesture_text == "fist":
                            sound_synth(0, self.mode, self.instrument_id)
                        elif keyboard.is_pressed("z"):
                            if not self.key_pressed:
                                sound_synth(
                                    self.midinote, self.mode, self.instrument_id
                                )
                                self.key_pressed = True
                        if not keyboard.is_pressed("z"):
                            self.key_pressed = False

        img = Image.fromarray(rgb_frame)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.root.after(10, self.update_video)


def sound_synth(midi_freq, mode, instrument_id):
    print(
        f"Sending Midinote Number: {midi_freq}, Mode: {mode}, Instrument: {instrument_id}"
    )
    client.send_message("/from_python", [midi_freq, mode, instrument_id])


if __name__ == "__main__":
    root = tk.Tk()
    app = HandGestureApp(root)
    root.mainloop()
