# -*- coding: utf-8 -*-
"""ProjectML6375.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1PnE93IU-B3P3qs8AdNy93imsLEhDh8Cl
"""

print('Installing dependencies...')
!apt-get update -qq && apt-get install -qq libfluidsynth1 fluid-soundfont-gm build-essential libasound2-dev libjack-dev
!pip install -qU pyfluidsynth pretty_midi

!pip install -qU magenta

# Hack to allow python to pick up the newly-installed fluidsynth lib. 
# This is only needed for the hosted Colab environment.
import ctypes.util
orig_ctypes_util_find_library = ctypes.util.find_library
def proxy_find_library(lib):
  if lib == 'fluidsynth':
    return 'libfluidsynth.so.1'
  else:
    return orig_ctypes_util_find_library(lib)
ctypes.util.find_library = proxy_find_library

print('Importing libraries and defining some helper functions...')
from google.colab import files

import magenta
import note_seq
import tensorflow

print('🎉 Done!')
print(magenta.__version__)
print(tensorflow.__version__)

note_seq.notebook_utils.download_bundle('basic_rnn.mag', '/content/')
note_seq.notebook_utils.download_bundle('attention_rnn.mag', '/content/')
note_seq.notebook_utils.download_bundle('lookback_rnn.mag', '/content/')
!sudo apt install timidity
!pip install scipy
!pip install matplotlib==3.1.3

import random

total = []

for i in range(-2 , 128):
    total.append(i)

#Generating list of random numbers for primer melody each time

for s in range(200):
  temp = []
  for i in range(6):
      temp.append(random.choice(total))

  !melody_rnn_generate \
  --config=basic_rnn \
  --bundle_file='basic_rnn.mag' \
  --output_dir='/content/basic' \
  --num_outputs=1 \
  --num_steps=128 \
  --primer_melody= temp

import os
  
# Function to rename multiple files
def main():
  
    for count, filename in enumerate(os.listdir("/content/basic/")):
        dst ="Basic_" + str(count) + ".mid"
        src ='/content/basic/'+ filename
        dst ='/content/basic/'+ dst
          
        # rename() function will
        # rename all the files
        os.rename(src, dst)
  
# Driver Code
if __name__ == '__main__':
      
    # Calling main() function
    main()

#For attention_rnn
import random
import os

total = []

for i in range(-2 , 128):
    total.append(i)
#Generating list of random numbers for primer melody each time

for s in range(200):
  temp = []
  for i in range(6):
      temp.append(random.choice(total))

  !melody_rnn_generate \
  --config=attention_rnn \
  --bundle_file='attention_rnn.mag' \
  --output_dir='/content/attention' \
  --num_outputs=1 \
  --num_steps=128 \
  --primer_melody= temp


  
# Function to rename multiple files
def main():
  
    for count, filename in enumerate(os.listdir("/content/attention/")):
        dst ="Attention_" + str(count) + ".mid"
        src ='/content/attention/'+ filename
        dst ='/content/attention/'+ dst
          
        # rename() function will
        # rename all the files
        os.rename(src, dst)
  
# Driver Code
if __name__ == '__main__':
      
    # Calling main() function
    main()




#for lookback_rnn
import random
import os

total = []

for i in range(-2 , 128):
    total.append(i)
#Generating list of random numbers for primer melody each time

for s in range(200):
  temp = []
  for i in range(6):
      temp.append(random.choice(total))

  !melody_rnn_generate \
  --config=lookback_rnn \
  --bundle_file='lookback_rnn.mag' \
  --output_dir='/content/lookback' \
  --num_outputs=1 \
  --num_steps=128 \
  --primer_melody= temp


  
# Function to rename multiple files
def main():
  
    for count, filename in enumerate(os.listdir("/content/lookback/")):
        dst ="Lookback_" + str(count) + ".mid"
        src ='/content/lookback/'+ filename
        dst ='/content/lookback/'+ dst
          
        # rename() function will
        # rename all the files
        os.rename(src, dst)
  
# Driver Code
if __name__ == '__main__':
      
    # Calling main() function
    main()

!zip -r /content/file2_basic.zip /content/basic

!zip -r /content/file1_attention.zip /content/attention

!zip -r /content/file3_lookback.zip /content/lookback

