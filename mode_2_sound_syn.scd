// Start the OSC listener for /playfile
OSCdef(
	\playfile,  // Define the OSC listener name
	{ |msg|
		var filePath = msg[1];  // Get the file path from the OSC message

		// Load the audio file into a buffer
		var buf = Buffer.read(s, filePath, action: {
			// Play the buffer when it's loaded
			{
				var signal = PlayBuf.ar(buf.numChannels, buf, BufRateScale.kr(buf), loop: 1);
				Out.ar(0, signal );  // Adjust the volume as needed
			}.play;
		});
	},
	"/playfile"  // OSC message address
);