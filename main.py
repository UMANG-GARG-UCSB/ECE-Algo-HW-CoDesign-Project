import numpy as np
import torch
import itertools
import scipy
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm

from pydub import AudioSegment

# datapath = 'C:\/Users\/Yu\/Desktop\/train\/'
datapath = 'D:\/Study_Files\/UCSB\/Courses\/ECE 594BB Hardware for AI\/ProjectWorkSpace\/train\/'

torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_printoptions(
    precision=4,
    sci_mode=False
)

np.set_printoptions(suppress=True)


# Reproducity;
np.random.seed(1137)
torch.manual_seed(114514)

def readaudiodata(audiopath, frame_rate=8000, normalized=False):
    audio = AudioSegment.from_file(audiopath, format='m4a')
    audio = audio.set_frame_rate(frame_rate)
    audio_array = np.array(audio.get_array_of_samples())
    # Get only one channel data;
    if audio.channels == 2:
        audio_array = audio_array.reshape((-1, 2))[:, 0]
    if normalized:
        return audio.frame_rate, np.float32(audio_array) / 2**15
    else:
        return audio.frame_rate, audio_array

def cutframe(audiodata, frame_length=256, frame_overlap=128):
    frame_move = frame_length - frame_overlap
    audiolength = len(audiodata)
    num_frame = int(np.ceil((audiolength - frame_overlap) / frame_move))
    audio_frames = np.zeros([num_frame, frame_length])
    # Add zero pads for the last frame if needed;
    pad_length = int((num_frame-1)*frame_move+frame_length) - audiolength
    if (pad_length > 0):
        pad = np.zeros(pad_length)
        pad_audiodata = np.concatenate((audiodata, pad))
    else:
        pad_audiodata = audiodata
    for i in range(num_frame):
        audio_frames[i] = pad_audiodata[i*frame_move:i*frame_move+frame_length]
    return audio_frames

def addhanningwindow(audio_frame):
    # Add Hanning window to the framed data;
    num_frame = audio_frame.shape[0]
    frame_length = audio_frame.shape[1]
    hanningframe = np.zeros([num_frame, frame_length])
    hanningwindow = np.hanning(frame_length)
    for i in range(num_frame):
        hanningframe[i] = audio_frame[i] * hanningwindow
    return hanningframe


# Load training data;
max_data_length = 8192
max_data_length_real = 0

# Raw data, containing cat, apple and box;
raw_data = np.zeros([3, 5, max_data_length])

# Load raw data;
index = 0
name_list = list(['cat', 'apple', 'box'])
for name in name_list:
    for i in range(5):
        filepath = datapath + name + '{}'.format(i + 1) + '.m4a'
        sr, data = readaudiodata(filepath, frame_rate=8000)
        max_data = np.amax(data)
        data = data / max_data
        raw_data[index, i, 0:len(data)] = data
        # Find max data length;
        if (len(data) > max_data_length_real):
            max_data_length_real = len(data)
    index += 1

print(max_data_length_real)

print(raw_data.shape)

cat_raw_data = raw_data[0]
apple_raw_data = raw_data[1]
box_raw_data = raw_data[2]


# Alignment of raw data;
# Align the data and use only 4096 points;
raw_data_new = np.zeros([3, 5, 4096])
print(raw_data_new.shape)
initial_point = 0
for i in range(3):
    for j in range(5):
        for k in range(max_data_length):
            if (np.abs(raw_data[i, j, k]) > 0.03):
                initial_point = k
                break
        for k in range(max_data_length - initial_point):
            raw_data[i, j, k] = raw_data[i, j, k + initial_point]
        raw_data_new[i, j] = raw_data[i, j, 0:4096]


# Plot the fft of the whole sequence with 8192 point fft;
fft_point = 4096
raw_fft = np.zeros([3, 5, int(fft_point/2)])
for i in range(3):
    plt.figure(i + 1, figsize=(19.2, 9.6))
    for j in range(5):
        raw_fft[i, j, :] = np.abs(np.fft.fft(raw_data_new[i, j, :], fft_point))[0:int(fft_point/2)]
        plt.subplot(5, 1, j+1)
        plt.plot(raw_fft[i, j, :])
    plt.suptitle('{} point fft of {} data'.format(fft_point, name_list[i]))
    plt.savefig(datapath + 'plot\/' + 'fft_{}'.format(name_list[i]) + '.png')

# Plot stft of the input data, frame size: 32, no overlap, frame length;
stft_point = 256
frame_size = int(4096/stft_point)
framed_data = np.zeros([3, 5, frame_size, stft_point])
final_data = np.zeros([3, 5, frame_size, stft_point])
stft_data = np.zeros([3, 5, frame_size, int(stft_point/2)])

for i in range(3):
    for j in range(5):
        framed_data[i, j] = cutframe(raw_data_new[i, j], frame_length=stft_point, frame_overlap=0)
        # final_data[i, j] = addhanningwindow(framed_data[i, j])
        final_data = copy.deepcopy(framed_data)
        for k in range (frame_size):
            stft_data[i, j, k] = np.abs(np.real(np.fft.fft(final_data[i, j, k], stft_point)))[0:int(stft_point/2)]

# Unroll the frame;
stft_data_unroll = stft_data.reshape([3, 5, 1, -1])
for i in range(3):
    plt.figure(3 + i + 1, figsize=(19.2, 9.6))
    for j in range(5):
        plt.subplot(5, 1, j+1)
        plt.plot(stft_data_unroll[i, j, 0, :])
    plt.suptitle('{} point stft of {} data'.format(stft_point, name_list[i]))
    plt.savefig(datapath + 'plot\/' +  'stft_{}'.format(name_list[i]) + '.png')

# Feature extraction;
# Find the max amplitude of each frame;
stft_data_max_unroll = np.amax(stft_data, axis=3)
# Find the frequency with maximum amplitude in each frame;
# stft_data_max_unroll = np.argmax(stft_data, axis=3)
# Find the max frequencies across all frames;
# stft_data_max_unroll = np.argmax(stft_data, axis=2)

for i in range(3):
    plt.figure(6 + i + 1, figsize=(19.2, 9.6))
    for j in range(5):
        plt.subplot(5, 1, j+1)
        plt.stem(stft_data_max_unroll[i, j])
    plt.suptitle('Max stft for each frame of {} data'.format(name_list[i]))
    plt.savefig(datapath + 'plot\/' +  'stft_max_{}'.format(name_list[i]) + '.png')

# Raw data plot as a reference;
for i in range(3):
    plt.figure(9 + i + 1, figsize=(19.2, 9.6))
    for j in range(5):
        plt.subplot(5, 1, j+1)
        plt.plot(raw_data_new[i, j])
    plt.suptitle('Raw data in time domain of {} data'.format(name_list[i]))
    plt.savefig(datapath + 'plot\/' +  'raw_{}'.format(name_list[i]) + '.png')


# Save raw data;
np.save(datapath + 'raw_data.npy', raw_data_new)

# Save stft data;
np.save(datapath + 'stftdata.npy', stft_data_max_unroll)

