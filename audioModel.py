#%% 
from mmsdk import mmdatasdk
import os

audio_file = './data/cmumosei/CMU_MOSEI_COVAREP.csd'

data = mmdatasdk.mmdataset({'audio_feature': audio_file})
# %%

label_file = './data/cmumosei/CMU_MOSEI_Labels.csd'
data.add_computational_sequences({'label': label_file}, destination=None)
data.align('label')
# %%

# Extract video IDs out of the keys

import re
import numpy as np

pattern = re.compile('(.*)\[.*\]')

data_array = []

for segment in data['label'].keys():
    video_ID = re.search(pattern, segment).group(1)
    label = data['label'][segment]['features']
    audio = data['audio_feature'][segment]['features']

    # remove NAN values
    label = np.nan_to_num(label).flatten()
    audio = np.nan_to_num(audio).flatten()

    data_array.append((audio, label, segment))

# %%

def multi_collate(batch):
    '''
    Collate functions assume batch = [Dataset[i] for i in index_set]
    '''
    # for later use we sort the batch in descending order of length
    batch = sorted(batch, key=lambda x: x[0][0].shape[0], reverse=True)
    
    # get the data out of the batch - use pad sequence util functions from PyTorch to pad things
    labels = torch.cat([torch.from_numpy(sample[1]) for sample in batch], dim=0)
    sentences = pad_sequence([torch.LongTensor(sample[0][0]) for sample in batch], padding_value=PAD)
    visual = pad_sequence([torch.FloatTensor(sample[0][1]) for sample in batch])
    acoustic = pad_sequence([torch.FloatTensor(sample[0][2]) for sample in batch])
    
    # lengths are useful later in using RNNs
    lengths = torch.LongTensor([sample[0][0].shape[0] for sample in batch])
    return sentences, visual, acoustic, labels, lengths


# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)

# %%




# %%
