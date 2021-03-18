# SpeakerRecognition

## Intended Use
The intention is that this repository can be accessed by anyone and run on their own machine provided that all of the files have been downloaded properly.
The main program used here is speaker_predict_DylanShadduck.m
Other matlab files in this repository were used to generate the method used in speaker_predict_DylanShadduck.m which is now my third iteration of this project. 

```
% Example Usage of the functions in speaker_predict_DylanShadduck.m

% Define the locations of the train and test folders
train_folder = "PATH\TO\TRAIN\FOLDER";
test_folder = "PATH\TO\TEST\FOLDER";

% Set the seed of random number generation for completely repeatable results
rng("default");

% Set blocking parameters
N = 256;
M = 100;

% Set mel filter number
K = 20;

% Set the number of clusters for kmeans clustering 
% My testing shows three or four clusters perform quite well
clusters = 3;

% Train the model
C = train_model(train_folder, N, M, K, clusters);

% Predict which speaker in the test dataset corresponds to the training dataset
predict_speaker(C, test_folder, train_folder, N, M, K);

% Generate the plots
generate_plots(train_folder)';
```

## Function Definitions

frame_block(data, N, M)
* Inputs:
  * data: A 1D vector representing the audio data to be frame blocked
  * N: A scalar integer that represents the size of each frame
  * M: A scalar integer that represents the number of elements that will overlap between frames
* Outputs:
  *  B: A 2D array with each row representing a frame of the original input data. The number of rows is dependent on how large the input data vector is

```
% Example usage
[y, fs] = audioread(audiofilename);

N = 256;
M = 100;

blocked_array = frame_block(y, N, M);
```

These values will result in a blocked array that has 256 columns where each frame overlaps with the next by 100 samples

matrix_hamm(block_matrix)
* Inputs:
  * block_matrix: A 2D array in blocked form where there are N number of columns representing the number of samples in a single frame
*  Outputs:
  *  B: A 2D block matrix where each row has had a hamming window applied to it

```
% Example usage
blocked_array = frame_block(y, N, M);

windowed_block = matrix_hamm(blocked_array);
```

Where windowed_block is a 2D array with N columns with a hamming window applied to each row of blocked_array

mel_points(freq_min, freq_max, mel_filt_num, fft_size, samp_freq)
* Inputs:
  * freq_min: The minimum frequency for mel spaced filter (typically zero)
  * freq_max: The maximum frequency for the mel spaced filter bank (typically half the sample frequency)
  * mel_filt_num: The number of mel spaced filters to use
  * fft_size: The length of the fft used to find the frequency response of each frame
  * samp_freq: The sample frequency of the data we will be applying the mel frequency filter to
* Outputs:
  * points: These are the different sampling points from 1 to fft_size where the center of each filter will be
  * freqs: These are the frequencies that correspond to the sample points above

```
% Example Usage
[y, fs] = audioread(filename);

[points, freqs] = mel_points(0, fs/2, N, fs)
```

mel_bank(samp_freq, fft_size, mel_filt_num, norm)
* Inputs:
  * samp_freq: The sampling frequency of the data we will be applying these filters to
  * fft_size: The size of the fft done on each frame of the audio data
  * mel_filt_num: The number of filters to use
  * norm: A boolean (true or false) variable that will normalize the area of each triangular filter when set to true. The default usage is true
* Outputs:
  * m: The mel spaced filter bank

```
% Example usage
[y, fs] = audioread(filename);
N = 256;
M = 100;

blocked_array = frame_block(y, N, M);
windowed_block = matrix_hamm(blocked_array);

% Taking the fft of the windowed block and finding the power
fft_block = fft(windowed_block, 2);
power_fft = abs(fft_block).^2;

K = 20;
m = mel_bank(fs, N, K, true);

% Filtering our power fft blocked array
y_filtered = m * power_fft;
```

get_mfcc(data, samp_freq, fft_length, overlap_length, mel_filt_num, show_plot)
```
% This function takes sampled audio data and returns the Mel-Frequency-Cepstrum-Coefficients 2D array
```

* Inputs:
  * data: The sampled audio data
  * samp_freq: Sampling frequency of data
  * fft_length: The number of samples in a framed block 
  * mel_filt_num: The number of filters to use for the mel spaced filter bank
  * show_plot: A boolean variable that will show some plots when true. The default use is false
* Outputs:
  * mfcc_coeff: The mel frequency cepstrum coefficients of the input data

```
% Example Usage
[y, fs] = audioread(filename);

% Defining fft size, overlap length, and the number of mel filters
N = 256;
M = 100;
K = 20;

mfcc = get_mfcc(y, fs, N, M, K, false);
```

trim_silence(audio_data, min_volume)
* Inputs:
  * audio_data: Sampled audio data
  * min_volume: The threshold volume that will determine the beginning and the end of the trimming
* Outputs:
  * y: The trimmed audio data

```
% Example usage
[y, fs] = audioread(filename);

% It's important to normalize the data before trimming
y = y./max(abs(y));

% Set the normalized minimum volume
min_volume = 0.05;

y_trimmed = trim_silence(y, min_volume);
```

train_model(train_folder, fft_size, overlap_len, mel_filt_num, num_clusters)
* Inputs:
  * train_folder: A string representing the path to the training folder 
  * fft_size: The size of the frames for frame blocking
  * overlap_len: The number of samples that will overlap in each frame during frame blockinng
  * mel_filt_num: The number of mel spaced filters to use
  * num_clusters: The number of clusters used for kmeans clustering
* Outputs:
  * C: An N-D array that has centroids of each cluster for each user. Shape is (num_speakers, num_clusters, mel_filt_num-1)

```
% Example Usage
train_folder = "PATH TO TRAIN FOLDER";

% Set our blocking parameters
N = 256;
M = 100;

% Set our mel filter number
K = 20;

% Set our number of clusters
clusters = 3;

C = train_model(train_folder, N, M, K, clusters);
```

predict_speaker(trained_centroids, test_folder, train_folder, fft_size, overlap_len, mel_filt_num)
* Inputs:
  * trained_centroids: N-D array of all the centroids for each cluster for each user. Typically comes from the train_model function above
  * test_foler: String with the path to the training folder
  * train_folder: String with the path to the training folder
  * fft_size: The size of the frames for frame blocking
  * overlap_len: The number of samples that will overlap in each frame during frame blockinng
  * mel_filt_num: The number of mel spaced filters to use
* Outputs:
  * This function prints to the stdout

```
% Example Usage
train_folder = "Path to train folder"
test_folder = "Path to test folder"

% Blocking parameters
N = 256;
M = 100;
K = 20;
clusters = 3;

% Train the model and find the centroids
C = train_model(train_folder, N, M, K, clusters);

% Make the predictions for our test data set
predict_speaker(C, test_folder, train_folder, N, M, K);
```

generate_plots(train_folder)
```
% This function generates all of the required plots for the training data set
```
* Inputs:
  * train_folder: String with the path to the training folder
* Outputs:
  * This function only generates matlab figures

```
% Example Usage
train_folder = "Path to train data";

generate_plots(train_folder);
```
