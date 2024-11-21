# Guide for Using

## Terminal Version

### Running the program

1. Open `sound_syn.scd` in SuperCollider.
2. Boot the server, can use `ctrl/cmd + B`.
3. Evaluate file or session, can use `ctrl/cmd + enter`.
4. Open a terminal and navigate to the directory containing `terminal_version.py`</mark>`.
5. Run the script.

### Using the Program

- The program will start capturing video from your webcam.
- You will see the current mode and instrument displayed on the video feed.

### Modes

- **Frequency Mode**: The frequency of the sound is controlled by the vertical position of your hand.
- **Discrete Midi Mode**: The MIDI note number is controlled by the vertical position of your hand. **_Remember to press `s` to reset position first_**.

### Instruments

- Sine oscillator
- Hand flute

### Controls

- Press `m` to switch between Frequency Mode and Discrete Midi Mode.
- Press `i` to switch between instruments.
- Press `s` to reset the position for MIDI note calculation.
- Press `z` to play the sound based on the detected gesture.
- Press `q` to quit the program.

### Gestures

- `Fist`: Plays a sound with frequency 0 or stops the sound.
- `Open Hand`: Plays a sound based on the current mode and hand position.

### Example Usage

1. Start the program.
2. Press `m` to switch to Discrete Midi Mode.
3. Press `s` to reset position.
4. Press `i` to switch to Hand flute.
5. Make a fist to stop the sound.
6. Open your hand and move it up or down to change the MIDI note.
7. Press `z` to play the sound.
8. Press `q` to quit the program.

## GUI Version

### Running the program

1. Open `sound_syn.scd` in SuperCollider.
2. Boot the server, can use `ctrl/cmd + B`.
3. Evaluate file or session, can use `ctrl/cmd + enter`.
4. Open a terminal and navigate to the directory containing `gui_version.py`</mark>`.
5. Run the script.

### Using the Program

- The program will open a GUI window with the title "Hand Gesture Music Controller".
- The GUI will display the video feed from your webcam and information about the current mode and instrument.

### Modes

- **Frequency Mode**: The frequency of the sound is controlled by the vertical position of your hand.
- **Discrete Midi Mode**: The MIDI note number is controlled by the vertical position of your hand. **_Remember to press `s` to reset position first_**.

### Instruments

- **Sine oscillator**
- **Hand flute**

### Controls

- Click `Change Mode` button to switch between Frequency Mode and Discrete Midi Mode.
- Click `Change Instrument` button to switch between instruments.
- Click `Reset Position` button to reset the position for MIDI note calculation.
- Click top right cross button to close window.
- working...

### Gestures

- `Fist`: Plays a sound with frequency 0 or stops the sound.
- `Open Hand`: Plays a sound based on the current mode and hand position.

### Example Usage

working...

# Log

1. Python `keyboard` library is not fully working on MacOS. For MacOS, it is suggested to use `pynput` library or `cv2.waitKey` function from `OpenCV`.

2. **Idea**: Use hand to control the music generation. Music is generated by SuperCollider. Hand controls the parameters which is sent to the SuperCollider server.

3. Hand controls frequency in frequency mode, and midi note in Discrete Midi Mode.

4. - Use the thumb and index thinger to replace the "z" key function in terminal version.
   - Change the GUI to PyGame from tkinter
   - Add drum, chord, etc, by other hand gestures
   - if time allowed, add graphics
   - 可以疊加音軌，不同手勢啟動不同音軌（drum）。手的高度控制音軌速度或者頻率，然後用一個手勢來確定。

5. Further implementation. If hand can control it, voice can also can as it is the same principle as recognition.
