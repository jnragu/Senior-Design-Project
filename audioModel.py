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

    data_array.append((audio, label))

# %%

import pandas as pd

data_df = pd.DataFrame(data_array, columns=['audio', 'label'])

target = data_df.pop('label')

# %%
import tensorflow as tf
data_tf = tf.data.Dataset.from_tensor_slices((data_df.values, target.values))



# %%
