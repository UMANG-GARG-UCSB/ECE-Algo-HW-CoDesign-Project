import numpy as np

np.set_printoptions(suppress=True)
# datapath = 'C:\/Users\/Yu\/Desktop\/train\/'
datapath = 'D:\/Study_Files\/UCSB\/Courses\/ECE 594BB Hardware for AI\/ProjectWorkSpace\/train\/'

# Number of fraction bits;
Q_bit = 8
# Number type;
dtype = np.int32

def np_relu (a, dtype_=dtype):
    dim = a.shape[0]
    b = np.zeros(dim, dtype=dtype_)
    for i in range(dim):
        if (a[i] > 0): b[i] = a[i]
        else: b[i] = 0
    return  b

def np_crossentropy(a):
    return np.exp(a) / np.sum(np.exp(a))

# Return the converted int number of error;
def f2int(a, Q_bit_=Q_bit, dtype_=dtype):
    return dtype_(a * 2**Q_bit_), a - np.float32(dtype_(a * 2**Q_bit_)/2**Q_bit_)

def model(a):
    c0 = w1_f @ a + b1_f
    c1 = np_relu(c0, dtype_=np.float32)
    c2 = w2_f @ c1 + b2_f
    return c2

def model_int(a):
    c0 = dtype((w1_int @ a) / 2**Q_bit) + b1_int
    c1 = np_relu(c0)
    c2 = dtype((w2_int @ c1) / 2**Q_bit) + b2_int
    return np.float32(c2/2**Q_bit)

rawdata_f = np.load(datapath + 'raw_data.npy')

# [m][n][k] data, m: 0 = cat, 1 = apple, 1 = box;
# n: 5 different samples;
# k: feature;
data_f = np.load(datapath + 'stftdata.npy')
# Model parameter;
w1_f = np.load(datapath + 'w1.npy')
b1_f = np.load(datapath + 'b1.npy')
w2_f = np.load(datapath + 'w2.npy')
b2_f = np.load(datapath + 'b2.npy')

# Convert to desired type;
rawdata_int, _ = f2int(rawdata_f)
data_int, _ = f2int(data_f)
w1_int, _ = f2int(w1_f)
b1_int, _ = f2int(b1_f)
w2_int, _ = f2int(w2_f)
b2_int, _ = f2int(b2_f)

# Save the results;
np.save(datapath + 'data\/' + 'rawdata_int.npy', rawdata_int)
np.save(datapath + 'data\/' + 'data_int.npy', data_int)
np.save(datapath + 'data\/' + 'w1_int.npy', w1_int)
np.save(datapath + 'data\/' + 'b1_int.npy', b1_int)
np.save(datapath + 'data\/' + 'w2_int.npy', w2_int)
np.save(datapath + 'data\/' + 'b2_int.npy', b2_int)

# print(model(data_f[0][4]))
# print(model_int(data_int[0][4]))

test_result = []
real_result = []
test_score = np.zeros([3, 5, 3])
real_score = np.zeros([3, 5, 3])
for i in range(3):
    for j in range(5):
        t_score = np_crossentropy(model_int(data_int[i][j]))
        r_score = np_crossentropy(model(data_f[i][j]))
        test_score[i][j] = t_score
        real_score[i][j] = r_score
        test_result.append(np.argmax(t_score, axis=0))
        real_result.append(np.argmax(r_score, axis=0))

print('Ideal classification result is:\n', real_result)

print('Result for {} fraction bits, with format {} is:\n'.format(Q_bit, dtype), test_result)

# print('Real score is:\n', real_score)
#
# print('Test score is:\n', test_score)