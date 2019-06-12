'''
Contains functions required by sed_main.ipynb.

Created for the purpose of sound event detection.
by Reynaldo Vazquez (rexvaz.com)
version June 5, 2019

Licensed under CC BY 4.0 (http://creativecommons.org/licenses/by/4.0/)
'''

import os, sys, io, h5py, IPython
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import soundfile, librosa
import librosa.display
from scipy.io import wavfile
from tabulate import tabulate
from io import BytesIO
from zipfile import ZipFile
from pydub import AudioSegment
AudioSegment.converter =  '/anaconda3/ffmpeg'

from __main__ import *

mpl.rcParams.update({'text.color' : "#4d4d4d", 'axes.labelcolor' : "#4d4d4d",
                     'font.size': 13, 'xtick.color':'#4d4d4d','ytick.color':'#4d4d4d',
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#a6a6a6', 'axes.linewidth':1.0, 'figure.figsize':[8, 4]})

def get_wav_info_from_zip(path_in_archive, archive, target_fs = 44100):
    '''
    Extracts data from audio file in .wav format contained in a zip
    archive. Standardizes to 1 channel and to a sample rate of 44.1KHz.
    '''
    audio_zip = archive.read(path_in_archive)
    bytes_io = BytesIO(audio_zip)
    rate, data = wavfile.read(bytes_io)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    if target_fs is not None and rate != target_fs:
        data = librosa.resample(data, orig_sr=rate, target_sr=target_fs)
        rate = target_fs
    return rate, data

def log_mel_features(audio_data, sr, mel_power):
    '''
    Extracts the log mel features from audio data.
    '''
    S = librosa.feature.melspectrogram(y=audio_data, sr=sr)
    lmf = librosa.power_to_db(S**mel_power, ref=np.max)
    return lmf

def plot_log_mel(log_mel, mel_id=None, sr = 44100):
    '''
    Plots the log mel spectrogram
    '''
    plt.box(False)
    librosa.display.specshow(log_mel, cmap='Blues_r',
                             x_axis='time', sr = sr)
    cb = plt.colorbar(format='%+2.0f dB')
    cb.outline.set_visible(False)
    plt.title('log mel spectrogram')
    plt.locator_params(axis='x', nbins=10)
    plt.xlim(0,10)
    return

def show_audio_info(archive, track_num, meta_segments, prefix):
    '''
    Displays the meta information, sound waves, mel spectrogram and play widget,
    of an audio file  contained in a zip archive.
    '''
    print('Audio clip id: ', str(track_num))
    path_in_archive = prefix+str(track_num)+'.wav'
    rate, data = get_wav_info_from_zip(path_in_archive, archive)
    logmel = log_mel_features(data, rate, mel_power=0.5)
    audio_zip = archive.read(path_in_archive)
    plt.figure(figsize=(17, 3))
    plt.subplot(1,2,2)
    plot_log_mel(logmel, track_num, rate)
    plt.subplot(1,2,1)
    librosa.display.waveplot(data, sr=44100)
    plt.title('sound waves')
    plt.locator_params(axis='x', nbins=10)
    plt.show()

    events = meta_segments[meta_segments['trackID'] == track_num].sort_values('event_start')
    if len(events) == 0:
        print('No audio events in this track, only background')
    else:
        table_keys=['event type', 'start time (sec)', 'end time (sec)']
        event_df = pd.concat([events['event'], round(events['event_start']/1000,2),
                 round(events['event_end']/1000,2)], axis=1, keys=table_keys)
        print(tabulate(event_df, headers='keys', showindex=False))

    bytes_io = BytesIO(audio_zip)
    old = AudioSegment.from_wav(bytes_io)
    old.export('media/temp_mp3.mp3', format='mp3')
    return IPython.display.Audio('media/temp_mp3.mp3')

def dataset_totals(meta_track, dataset_type):
    '''
    Calculates prints audio distribution counts using meta data
    '''
    number_of_tracks = meta_track.shape[0]
    num_positive_tracks = sum(meta_track['class_dummy'] == 1)
    num_non_positive_tracks = sum(meta_track['class_dummy'] == 0)
    print('Totals for ', dataset_type, ' dataset' )
    print('number of tracks:', number_of_tracks)
    print('number of tracks with positive event occurrence:', num_positive_tracks)
    print('number of tracks without positive event occurrence:', num_non_positive_tracks, '\n')
    return number_of_tracks, num_positive_tracks, num_non_positive_tracks

def make_bar_plot(x_names, y_values, color, width = 0.8):
    '''
    Make a bar plot
    '''
    xpos = np.arange(len(x_names))
    plt.bar(xpos, y_values, align='center', alpha=0.7, color = color, width=width)
    plt.xticks(xpos, x_names)

def tracks_content_info(meta_track, meta_segments, dataset_type, track_dur, event_type):
    '''
    Calculates the audio content on the dataset using meta data
    '''
    total_audio_time_sec = meta_track.shape[0]*track_dur
    total_audio_time_min = round(total_audio_time_sec/60, 2)

    meta_segments['event_dur'] = meta_segments['event_end'] - meta_segments['event_start']
    total_time_event_ms = sum(meta_segments['event_dur'][meta_segments['event'] == event_type])
    total_time_event_min = round(total_time_event_ms/(60*1000), 2)

    total_time_neg_ms = sum(meta_segments['event_dur'][meta_segments['event'] != event_type])
    total_time_neg_min = round(total_time_neg_ms/(60*1000), 2)

    total_time_bgd_only_min = round(total_audio_time_min - total_time_event_min - total_time_neg_min,2)
    print('Audio contents of', dataset_type, ' dataset' )
    print('Total audio minutes:', total_audio_time_min)
    print('Total glassbreak minutes:', total_time_event_min)
    print('Total gunshot minutes:', total_time_neg_min)
    print('Total background only minutes:', total_time_bgd_only_min, '\n')
    return total_audio_time_min, total_time_bgd_only_min, total_time_event_min, total_time_neg_min

def distribution_plot(counts, x_names, y_lim, lab_offset1, lab_offset2, y_label = None, color = '#0059b3'):
    '''
    Creates audio distribution plots
    '''
    plt.subplots_adjust(top=.75)
    for i, name in enumerate(counts):
        plt.subplot(1,2,i+1)
        y_values = [counts[name][i] for i in range(1,len(counts[name]))]
        y_pct = [val/sum(y_values) for val in y_values]
        bar_lab = ['   '+str(format(y_values[j],'.0f'))+
                   '\n'+'('+str(format(y_pct[j]*100,'.1f'))+'%)' for j in range(len(y_values))]
        make_bar_plot(x_names, y_values, width = 0.55, color = color)
        plt.title(name + ' audioset (total '+ str(counts[name][0]) +')\n')
        plt.ylim(0,y_lim)
        plt.ylabel(y_label)
        for j, v in enumerate(y_values):
            plt.text(j-lab_offset1, v+lab_offset2, bar_lab[j])
    plt.show()
    return

# Part 2 - additions for ETL
def get_log_mel_features_from_zip(archive, meta, Tx, n_freq, target_fs, mel_power, prefix):
    '''
    Extracts log mel features from audio file in .wav format contained in a zip
    archive.
    '''
    mel_feats = None
    tracks_n = meta.shape[0]
    mel_feats = np.zeros((tracks_n, Tx, n_freq))
    for i, trackID in enumerate(meta.trackID):
        path = prefix+str(trackID)+'.wav'
        rate, data = get_wav_info_from_zip(path, archive, target_fs)
        xi = log_mel_features(data, rate, mel_power)
        xi = np.transpose(xi)
        mel_feats[i] = xi
    return mel_feats

def create_or_load_mel_features(zip_train, meta_train_track, zip_eval, meta_eval_track,
                                Tx, n_freq, sample_rate, mel_power, prefix_train,
                                prefix_eval):
    '''
    Extracts and saves mel features for all audio files contained in a zip
    archive. Loads them if the featues alerady exist.
    '''
    file_names = ['X_train_mel', 'X_eval_mel']
    if not os.path.exists('output_data'):
        os.makedirs('output_data')
        X_train_mel = get_log_mel_features_from_zip(zip_train, meta_train_track, Tx,
                                        n_freq, sample_rate, mel_power, prefix_train)
        X_eval_mel = get_log_mel_features_from_zip(zip_eval, meta_eval_track, Tx,
                                        n_freq, sample_rate, mel_power, prefix_eval)
        datasets_to_save = [X_train_mel, X_eval_mel]
        for i, dataset in enumerate(datasets_to_save):
            h5f = h5py.File('output_data/'+file_names[i]+'.h5', 'w')
            h5f.create_dataset('dataset', data=dataset)
            h5f.close()
    else:
        X_sets = []
        for i, name in enumerate(file_names):
            file_name = 'output_data/'+name+'.h5'
            h5f = h5py.File(file_name,'r')
            X_sets.append(h5f['dataset'][:])
            h5f.close()
        [X_train_mel, X_eval_mel] = X_sets
    return X_train_mel, X_eval_mel

# Part 3 - additions for Feature Engineering
def insert_ones(y, segment_start_ms, segment_end_ms, Ty, track_dur_ms, during, lag):
    """
    Update the label vector y at specified times.

    Arguments:
        y -- numpy array of shape (1, Ty), the labels of the training example
        segment_start_ms -- the start time of the segment in ms
        segment_end_ms -- the end time of the segment in ms
        during -- whether the labels are placed during the event (True), or after (False)
        lag -- number of lagged time frames

    Returns: y -- updated labels
    """
    segment_end_y = int(segment_end_ms * Ty / track_dur_ms)
    if during == False:
        for i in range(segment_end_y + 1, segment_end_y + lag + 1):
            if i < Ty:
                y[0, i] = 1
    else:
        segment_start_y = int(segment_start_ms * Ty / track_dur_ms)
        for i in range(segment_start_y + lag, segment_end_y + lag):
            if i < Ty:
                y[0, i] = 1
    return y

def create_labels(meta_track, meta_segments, Ty, track_dur_ms, during, lag, event_label):
    '''
    Creates a vector of labels for an audio track.
    '''
    labels = None
    tracks_n = meta_track.shape[0]
    labels = np.zeros((tracks_n, Ty, 1))

    for i, trackID in enumerate(meta_track.trackID):
        yi = np.zeros((1,Ty))
        if meta_track.class_dummy[i] == 1:
            sub = meta_segments.loc[meta_segments['trackID'] == trackID]
            for e, event in enumerate(sub['event']):
                if event == event_label:
                    segment_start_ms = sub['event_start'].iloc[e]
                    segment_end_ms = sub['event_end'].iloc[e]
                    yi = insert_ones(yi, segment_start_ms, segment_end_ms, Ty, track_dur_ms, during, lag)
        yi = np.transpose(yi)
        labels[i] = yi
    return labels

# Part 4 - additions for Model definition.
# Part 5 - additions for Model training
# Part 6 - additions for Model Evaluation
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score

# Functions to build the ground truth event dictionaries for train and eval sets
def make_instance_dictionary(file_name, onset, offset, scene_label, event_label):
    '''
    Makes a dictionary containing meta data for each event.
    '''
    d = {}
    d['event_label'] = event_label
    d['event_onset'] = onset
    d['event_offset'] = offset
    d['file'] = file_name
    d['scene_label'] = scene_label
    return d

def get_segments_ref(meta_track, meta_segments, event_label):
    '''
    Builds the ground truth event dictionaries for dev and eval sets
    '''
    segments_dicts_ref = []
    for i, trackID in enumerate(meta_track.trackID):
        if meta_track.class_dummy[i]:
            file_name = str(trackID) + '.wav'
            sub = meta_segments.loc[meta_segments['trackID'] == trackID]
            for e, event in enumerate(sub['event']):
                if event == event_label:
                    onset = sub['event_start'].iloc[e]
                    offset = sub['event_end'].iloc[e]
                    segment_dict = make_instance_dictionary(file_name, onset/1000,
                                            offset/1000, scene_label, event_label)
                    segments_dicts_ref.append(segment_dict)
    return segments_dicts_ref

# Functions to build the dictionaries of events predicted by the model.
def predicted_event_segments(pred_index, Y_pred):
    '''
    Extracts the indexes in which a positive prediction has been made
    '''
    pos_index = np.where(Y_pred[pred_index,:,0] == 1)[0]
    starts = [pos_index[0]]
    ends = []
    for t in range(len(pos_index)-1):
        if pos_index[t+1] - pos_index[t] > 1:
            ends.append(pos_index[t])
            starts.append(pos_index[t+1])
    ends.append(pos_index[len(pos_index)-1])
    return starts, ends
# For event based statistics (True Positives have no use or meaning)
def segments_dicts_est(predictions_matrix, meta_track, event_label = 'glassbreak', track_dur = 10.0, Ty = 212):
    '''
    Builds the dictionaries of all events predicted by the model
    predictions_matrix must be already turned into 0s and 1s by threshold,
    not raw predicitons
    '''
    lists_of_predicted_segments = []
    for t in range(predictions_matrix.shape[0]):
        if 1 in predictions_matrix[t,:,0]:
            starts, ends = predicted_event_segments(t, predictions_matrix)
            lists_of_predicted_segments.append([starts, ends])
        else:
            lists_of_predicted_segments.append(None)

    segments_dicts_est = []
    for p in range(len(predictions_matrix)):
        segments = lists_of_predicted_segments[p]
        if segments != None:
            file_name = str(meta_track.trackID[p]) + '.wav'
            for s in range(len(segments[0])):
                onset = (segments[0][s]-lag)*track_dur/Ty
                offset = (segments[1][s]-lag)*track_dur/Ty
                segment_dict = make_instance_dictionary(file_name, onset, offset,
                                                scene_label, event_label)
                segments_dicts_est.append(segment_dict)
    return segments_dicts_est

# Function to calculate Segment and Event based metrics
def event_based_metrics(segments_dicts_ref, segments_dicts_est,
                                t_col = 0.5, pct_len = 0.5, eval_offset = False):
    '''
    Calculates and displays event based metrics
    '''
    reference_event_list = dcase_util.containers.MetaDataContainer(segments_dicts_ref)
    estimated_event_list = dcase_util.containers.MetaDataContainer(segments_dicts_est)
    event_based_metrics = sed_eval.sound_event.EventBasedMetrics(
        event_label_list=reference_event_list.unique_event_labels,
        t_collar=t_col, percentage_of_length = pct_len, evaluate_offset=eval_offset,
        evaluate_onset=True, event_matching_type='optimal')
    for filename in reference_event_list.unique_files:
        reference_event_list_for_current_file = reference_event_list.filter(
            filename=filename)
        estimated_event_list_for_current_file = estimated_event_list.filter(
            filename=filename)
        event_based_metrics.evaluate(
            reference_event_list=reference_event_list_for_current_file,
            estimated_event_list=estimated_event_list_for_current_file)
    event_metrics = event_based_metrics.results_overall_metrics()
    error_rate_metrics_dict = event_metrics['error_rate']
    f_metrics_dict = event_metrics['f_measure']
    error_rate_metrics_df = pd.DataFrame.from_dict(error_rate_metrics_dict, orient='index')
    f_metrics_df = pd.DataFrame.from_dict(f_metrics_dict, orient='index')
    error_rate_metrics_df.index = error_rate_metrics_df.index.map(lambda x: x.replace('_', ' '))
    f_metrics_df.rename(index={'f_measure':'F1'},inplace=True)
    print('F-measure')
    print(tabulate(f_metrics_df, floatfmt='.2f', tablefmt='rst'), '\n')
    print('Error Rate')
    print(tabulate(error_rate_metrics_df, floatfmt='.2f', tablefmt='rst'))
    return event_metrics

# Function to return event based metrics tables
def ebm_tables(segments_dicts_ref, segments_dicts_est,
                                t_col = 0.5, pct_len = 0.5, eval_offset = False):
    '''
    Creates event based metrics tables
    '''
    reference_event_list = dcase_util.containers.MetaDataContainer(segments_dicts_ref)
    estimated_event_list = dcase_util.containers.MetaDataContainer(segments_dicts_est)
    event_based_metrics = sed_eval.sound_event.EventBasedMetrics(
        event_label_list=reference_event_list.unique_event_labels,
        t_collar=t_col, percentage_of_length = pct_len, evaluate_offset=eval_offset,
        evaluate_onset=True, event_matching_type='optimal')
    for filename in reference_event_list.unique_files:
        reference_event_list_for_current_file = reference_event_list.filter(
            filename=filename)
        estimated_event_list_for_current_file = estimated_event_list.filter(
            filename=filename)
        event_based_metrics.evaluate(
            reference_event_list=reference_event_list_for_current_file,
            estimated_event_list=estimated_event_list_for_current_file)
    event_metrics = event_based_metrics.results_overall_metrics()
    error_rate_metrics_dict = event_metrics['error_rate']
    f_metrics_dict = event_metrics['f_measure']
    error_rate_metrics_df = pd.DataFrame.from_dict(error_rate_metrics_dict, orient='index')
    f_metrics_df = pd.DataFrame.from_dict(f_metrics_dict, orient='index')
    error_rate_metrics_df.index = error_rate_metrics_df.index.map(lambda x: x.replace('_', ' '))
    f_metrics_df.rename(index={'f_measure':'f1-score'},inplace=True)
    print('F-measure')
    print(tabulate(f_metrics_df, floatfmt='.2f', tablefmt='rst'), '\n')
    print('Error Rate')
    print(tabulate(error_rate_metrics_df, floatfmt='.2f', tablefmt='rst'))
    return event_metrics, f_metrics_df, error_rate_metrics_df

x_axis_vals = np.arange(0,Ty)*track_dur/Ty

def plot_pred_vs_true(meta_track, actual_labels, raw_predictions, dataset,
                        zip_audioset, prefix, track_index, threshold = 0.50):
    '''
    Plots model predictions versus ground truth, displays audio play button.
    '''
    track_id = meta_track.trackID.iloc[track_index]
    y_test = actual_labels[track_index,:,0]
    y_pred = raw_predictions[track_index,:,0]
    title = dataset + ' clip id: '+str(track_id)
    fig, ax = plt.subplots(figsize=(12, 4))

    l2,=ax.plot(x_axis_vals, y_pred, color='#6699ff', alpha=0.4)
    ax.fill_between(x_axis_vals, 0, y_pred, color='#6699ff', alpha=0.4)
    l1,=ax.plot(x_axis_vals, y_test, color='#003399', linewidth=2.3)

    y_test_gs = actual_labels[track_index,:,1]
    y_pred_gs = raw_predictions[track_index,:,1]

    l4,=ax.plot(x_axis_vals, y_pred_gs, color='#ff4dd2', alpha=0.4)
    ax.fill_between(x_axis_vals, 0, y_pred_gs, color='#ff4dd2', alpha=0.4)
    l3,=ax.plot(x_axis_vals, y_test_gs, color='#800060', linewidth=2.3)

    ax.set_title(title, {'fontsize': 15}).set_color('#4d4d4d')
    ax.set_xlabel('time (s)', {'fontsize': 14}).set_color('#4d4d4d')
    ax.set_ylabel('probability', {'fontsize': 14}).set_color('#4d4d4d')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2, width=0.5, colors='0.4',labelsize = 12)
    leg = plt.legend([l1, l2, l3, l4],["glassbreak truth", "glassbreak prediction", "gunshot truth", "gunshot prediction"],
                     prop={'size': 12}, frameon=False, loc=(1.05,0.4))
    for text in leg.get_texts(): plt.setp(text, color = '#4d4d4d')
    for legthickness in leg.legendHandles: legthickness.set_linewidth(2.5)

    plt.ylim(0,1.05)
    plt.xlim(0,10)
    plt.locator_params(axis='x', nbins=10)
    fig.tight_layout()
    plt.show()
    audio_zip = zip_audioset.read(prefix+str(track_id)+'.wav')
    bytes_io = BytesIO(audio_zip)
    old = AudioSegment.from_wav(bytes_io)
    old.export('media/temp_mp3.mp3', format='mp3')
    IPython.display.display(IPython.display.Audio('media/temp_mp3.mp3'))
    return
