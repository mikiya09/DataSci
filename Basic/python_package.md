

### Basic 

#### &#x03e8; install from source 
```
# if some package cannot be installed because any reason 
# 1) install from source using 
# 2) use crepe as an example 
# 3) also work for dependencies, hmmlearn is the dependency for crepe 
# 4) first install dependencies from source, then target package 
>> git clone https://github.com/hmmlearn/hmmlearn.git
>> cd hmmlearn 
>> pip install . 
# do the same for crepe
```

#### &#x03e8; Sound 
```python
# play .wav file 
pip install simpleaudio

# code
import simpleaudio as sa
wave_obj = sa.WaveObject.from_wave_file("path/to/your/wav/file.wav")
play_obj = wave_obj.play()
play_obj.wait_done()
```

#### &#x03e8; Separate Melody and Accompaniment 
*[demucs](./song-separator)*
```python 

```


### Data Science
#### &#x03e8; Hidden Markov Chain
```python 
# for apple silicon to work with HMM
pip install pyhhmm 

# import 
from pyhhmm.gaussian import GaussianHMM
```

#### &#x03e8; Mongo DB 
```
# pip install
pip install pymongo
```

