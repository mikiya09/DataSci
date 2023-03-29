

### Basic 
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

