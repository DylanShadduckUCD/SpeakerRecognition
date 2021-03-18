# SpeakerRecognition

## Function Definitions

frame_block(data, N, M)
* Inputs:
  * data: A 1D vector representing the audio data to be frame blocked
  * N: A scalar integer that represents the size of each frame
  * M: A scalar integer that represents the number of elements that will overlap between frames
* Outputs:
  * B: A 2D array with each row representing a frame of the original input data. The number of rows is dependent on how large the input data vector is

```

% Example usage
[y, fs] = audioread(audiofilename);

N = 256;
M = 100;

blocked_array = frame_block(y, N, M);

```

These values will result in a blocked array that has 256 columns where each frame overlaps with the next by 100 samples

