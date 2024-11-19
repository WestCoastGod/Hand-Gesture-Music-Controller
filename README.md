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
