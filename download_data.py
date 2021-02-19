#%% 
from mmsdk import mmdatasdk

# High level contains the extracted features for each modality
cmumosei_highlevel = mmdatasdk.mmdataset(mmdatasdk.cmu_mosei.highlevel,'cmumosei/')

#%%

# Raw contains the raw transcripts, phenomes 
cmumosei_raw = mmdatasdk.mmdataset(mmdatasdk.cmu_mosei.raw, 'cmumosei/')


#%% 

# Labels
cmumosei_labels = mmdatasdk.mmdataset(mmdatasdk.cmu_mosei.labels, 'cmumosei/')
# %%
