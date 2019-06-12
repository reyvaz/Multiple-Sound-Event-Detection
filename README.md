<br>

# Multiple Sound Event Detection

Builds an algorithm that automatically detects the occurrence of two classes of
sound events and their onset.

A deep learning network with mixed Long Short-Term Memory (LSTM) and Gated
Recurrent Units (GRU), that uses transformations of the mel spectrogram,
achieves this with an event-based error rate of 0.1 and a F1-score of 0.94

Contents:

- [sed2_main.ipynb](sed2_main.ipynb) Notebook to run all algorithms and report
- [sed2_utils.py](sed2_utils.py) Contains functions required by the notebook above
- [meta](meta) Contains meta data for the audio files
- [sed2_trained.h5](sed2_trained.h5) The trained model in Keras
- [index.html](index.html) Edited html version of the `sed2_main.ipynb` notebook
- [html_config.py](html_config.py) Configuration settings for the html exporter
- [README.md](README.md) This file

The audio data created for this project can be found
[here](https://zenodo.org/record/3236975#.XPSRDdNKjgt)

[**See the rendered html report**](https://reyvaz.github.io/Multiple-Sound-Event-Detection)

[**See the rendered notebook**](https://nbviewer.jupyter.org/github/reyvaz/Multiple-Sound-Event-Detection/blob/master/sed2_main.ipynb)



<br>
