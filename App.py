# ============================================================================
# Name        : Signal Studio (Task1)
# Author      : Team (24)
# Version     : last version
# Members     : Misara Ahmed-Rawan Abdelrahman-Rewan Mohmed-Rahma Abdelkader
# Supervision : Dr/Tamer - Eng/ Mohamed Mostafa
# ============================================================================

# --------------------------------Imported Libraries--------------------------------#
from requests import session
from sqlalchemy import false
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import scipy
from scipy.signal import find_peaks
from scipy import signal
import time

# --------------------------------Page Configuration--------------------------------#
st.set_page_config(layout="wide", page_icon='💡', page_title="Signal Studio")
content_left, space1, content_middle, space2, content_right = st.columns([1.3, 0.1, 1, 0.1, 3])

# ----------------------------------Open css file------------------------------------#
with open("style.css") as css:
    st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)

# ------------------------------------Initialization-----------------------------------------#

Time = np.linspace(0, 3, 6000)
maxF = 1

# ------------------------------------Session states------------------------------------------#
if 'interactive' not in st.session_state or 'fig' not in st.session_state or 'Signal' not in st.session_state or 'Summation' not in st.session_state or 'start' not in st.session_state or 'frequency' not in st.session_state or 'amplitude' not in st.session_state or 'maxF' not in st.session_state or 'max_added' not in st.session_state:
    st.session_state.interactive = 0
    st.session_state.start = 0
    st.session_state.frequency = 4
    st.session_state.amplitude = 4
    st.session_state.Signal = dict()
    st.session_state.fig = go.Figure()
    st.session_state.Summation = []
    st.session_state.maxF = 0
    st.session_state.max_added=0
    st.session_state.freq = []
    st.session_state.amp = []

for i in st.session_state.Signal.keys():
    st.session_state.freq.append(st.session_state.Signal[i][0])
    st.session_state.amp.append(st.session_state.Signal[i][1])


#uploded
def Max_Freq_uploaded():
    if len(st.session_state.freq) == 0:
        st.session_state.maxF = 1
    else:
        st.session_state.maxF = max(st.session_state.freq)
        if (st.session_state.maxF>st.session_state.max_added):
            st.session_state.maxF=st.session_state.maxF
        else:
            st.session_state.maxF=st.session_state.max_added
            
    return st.session_state.maxF

# **********************************************************************************
# Function Name: sampling
# Parameters (in): None
# Parameters (out): None
# Return value: None
# Description: Sampling the given signal
# ***********************************************************************************

def sampling(fsample, t, sin):
    time_range = (max(t) - min(t))
    samp_rate = int(((len(t) / time_range)) / (fsample))
    #samp_rate = int((len(t) / time_range) / (fsample))
    if samp_rate == 0:
        samp_rate = 1
    samp_time = t[:: samp_rate]
    samp_amp = sin[:: samp_rate]
    return samp_time, samp_amp


# **********************************************************************************
# Function Name: sincInterpolation
# Parameters (in): fsample, t, sin
# Parameters (out): reconstucted_sig, samp_time, samp_amp
# Return value: reconstucted signal ,amplitude and time
# Description: Interpolating the sampled signal
# ***********************************************************************************

def sincInterpolation(fsample, t, sin):

    samp_time, samp_amp = sampling(fsample, t, sin)
    samp_time = np.array(samp_time)
    samp_amp = np.array(samp_amp)
    time_matrix = np.resize(t, (len(samp_time), len(t)))
    k = (time_matrix.T - samp_time) / (samp_time[1] - samp_time[0])
    resulted_matrix = samp_amp * np.sinc(k)
    reconstucted_sig = np.sum(resulted_matrix, axis=1)
    return reconstucted_sig, samp_time, samp_amp

# **********************************************************************************
# Function Name: download
# Parameters (in): time,signal
# Parameters (out): None
# Return value: csv
# Description: download the generated signal
# ***********************************************************************************

def download(time, signal, maxf):
    data = {"X_axis": time, "Y_axis": signal, 'max_freq': st.session_state.maxF}
    Download_csv = pd.DataFrame(data)
    Download_csv.to_csv('Downloaded_csv.csv', index=False)

# **********************************************************************************
# Function Name: generateSignal
# Parameters (in): None
# Parameters (out): None
# Return value: graph
# Description: generating signal and saving it
# ***********************************************************************************

def generateSignal():
    y = st.session_state.amplitude * np.cos(st.session_state.frequency * 2 * np.pi * Time)
    st.session_state.Signal.update({'Signal: ' + str(st.session_state.frequency) + ' hz, ' + str(
        st.session_state.amplitude) + ' mV': [st.session_state.frequency, st.session_state.amplitude, y]})

# **********************************************************************************
# Function Name: interactivePlot
# Parameters (in): amplitude,frequency
# Parameters (out): None
# Return value: graph
# Description: plotting signal without saving it
# ***********************************************************************************

def interactivePlot():
    if st.session_state.interactive == 1:
        y = st.session_state.amplitude * np.cos(st.session_state.frequency * 2 * np.pi * Time)
        signal_Summation = 0
        for i in st.session_state.Signal.keys():
            signal_Summation += st.session_state.Signal[i][2]
            st.session_state.Summation = signal_Summation
        signal_Summation += y
        st.session_state.fig = go.Figure()
        st.session_state.fig.add_trace(
            go.Scatter(visible=True, x=Time, y=signal_Summation, name="Interactive Plot")
        )
        st.session_state.interactive = 0
    else:
        signal_Summation = 0
        for i in st.session_state.Signal.keys():
            signal_Summation += st.session_state.Signal[i][2]
            st.session_state.Summation = signal_Summation
        st.session_state.fig = go.Figure()
        st.session_state.fig.add_trace(
            go.Scatter(visible=True, x=Time, y=st.session_state.Summation, name="Original Signal")
        )

# **********************************************************************************
# Function Name: updatePlot
# Parameters (in): None
# Parameters (out): None
# Return value: updating graph
# Description: updating any changes in signal
# ***********************************************************************************

def updatePlot():
    if len(st.session_state.Signal) == 0:
        st.session_state.fig = go.Figure()
    else:
        signal_Summation = 0
        for i in st.session_state.Signal.keys():
            signal_Summation += st.session_state.Signal[i][2]
            st.session_state.Summation = signal_Summation
        st.session_state.fig = go.Figure()
        st.session_state.fig.add_trace(
            go.Scatter(visible=True, x=Time, y=signal_Summation)
        )

# **********************************************************************************
# Function Name: Plot
# Parameters (in): None
# Parameters (out): None
# Return value:  graph
# Description: plotting graph
# ***********************************************************************************

def plot():
    st.session_state.fig.update_layout(
        autosize=True,
        width=400,
        height=830)
    st.session_state.fig.update_yaxes(title_text="Amplitude(mV)")
    st.session_state.fig.update_xaxes(title_text="Time(s)",)
    st.plotly_chart(st.session_state.fig, use_container_width=True)


# **********************************************************************************
# Function Name: addNoise
# Parameters (in): SNR
# Parameters (out): None
# Return value:  noised graph
# Description: adding noise on signal
# ***********************************************************************************

def addNoise(SNR):
    if len(st.session_state.Summation) == 0:
        st.write("")

    else:
        signal_watts = (st.session_state.Summation) ** 2
        sig_avg_watts = np.mean(signal_watts)
        sig_avg_db = 10 * np.log10(sig_avg_watts)
        noise_avg_db = sig_avg_db - SNR
        noise_avg_watts = 10 ** (noise_avg_db / 10)
        mean_noise = 0
        noise = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), len(signal_watts))
        st.session_state.Summation = st.session_state.Summation + noise
        st.session_state.fig.add_trace(
            go.Scatter(visible=True, x=Time, y=st.session_state.Summation, name="Noised Signal")
        )


# **********************************************************************************
# Function Name: mainPage
# Parameters (in): None
# Parameters (out): None
# Return value: None
# Description: includes uploading signal, generates, plotting , sampling, reconstruction, interpolation and adding noise
# ***********************************************************************************

def mainPage():
    with content_left:
        st.title("Signal Studio 📉")
        st.subheader("Upload signal")
        uploaded_file = st.file_uploader('upload', label_visibility="hidden")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            signal_uploaded = df["Y_axis"].values
            Time_uploaded = df["X_axis"].values

            st.session_state.Signal.update({'Signal: uploaded': [0, 0, signal_uploaded]})
            updatePlot()
            max_freq = df['max_freq'].values
            st.session_state.maxF = max_freq

        # ################################################################   New Design   ###################################################################################
        st.subheader("Add signal")

        frequency = st.slider('Frequency', 1.0, 100.0, step=0.5, on_change=interactivePlot())
        amplitude = st.slider('Amplitude', 1.0, 30.0, step=0.5, on_change=interactivePlot())
        if frequency and amplitude:
            if st.session_state.frequency == frequency and st.session_state.amplitude == amplitude:
                print("")
            else:
                st.session_state.frequency = frequency
                st.session_state.amplitude = amplitude
                st.session_state.interactive = 1
                interactivePlot()

        submitted = st.button("Add Signal")
        if submitted:
            generateSignal()
            updatePlot()

        def delete():
            if len(st.session_state.Signal) == 0:
                st.write("")

            else:
                st.session_state.Signal.pop(signal_To_Delete)
                if len(st.session_state.Signal) > 0:
                    updatePlot()
                else:
                    st.session_state.fig = go.Figure()

        signal_To_Delete = st.selectbox(label="Select signal to delete:", options=st.session_state.Signal)
        st.button(" Delete ⛔ ", on_click=delete)

    with content_middle:
        st.subheader("Options")
        target_snr_db = st.slider('SNR', 50.0, 0.0)
        sample = st.radio("Samplig", ["Normalized Frequency(Hz)", "FMax"])
        if sample == "Normalized Frequency(Hz)":
            sampling_frequency = st.slider('', 1, 100, key='key1')

        if sample == "FMax":
            Max_Freq_uploaded()
            if (st.session_state.max_added>st.session_state.maxF) :
                st.session_state.maxF = st.session_state.max_added
            else:
              st.session_state.maxF= st.session_state.maxF

            sampling_frequency_max = st.slider('', 1, 10, step=1)

            sampling_frequency=sampling_frequency_max*st.session_state.maxF


        add_noise = st.checkbox("Show Noise")
        sample = st.checkbox("Show Sampling")
        reconstruct = st.checkbox("Show Reconstruction")

    with content_right:

        if uploaded_file is not None:

            if add_noise:
                addNoise(target_snr_db)

            # sampling_time, sampled_signal = sampling(sampling_frequency, Time_uploaded, signal_uploaded)
            if sample:

                # -------------------------------------------- Sampling --------------------------------------------------------------#
                if add_noise:
                    sampling_time, sampled_signal = sampling(sampling_frequency, Time_uploaded,
                                                             st.session_state.Summation)
                    st.session_state.fig.add_trace(
                        go.Scatter(visible=True, x=sampling_time, y=sampled_signal, mode='markers',
                                   name="Sampling With Noise")
                    )

                else:
                    sampling_time, sampled_signal = sampling(sampling_frequency, Time_uploaded,
                                                             st.session_state.Summation)
                    st.session_state.fig.add_trace(
                        go.Scatter(visible=True, x=sampling_time, y=sampled_signal, mode='markers', name="Sampling")
                    )

            if reconstruct:
                # ------------------------------------------ Reconstruction ---------------------------------------------------------------#

                reconstructed_signal, sam_time, sam_amp = sincInterpolation(sampling_frequency, Time_uploaded,
                                                                            st.session_state.Summation)
                st.session_state.fig.add_trace(
                    go.Scatter(visible=True, x=Time_uploaded, y=reconstructed_signal, name="Reconstruction")
                )
        else:
            if add_noise:
                addNoise(target_snr_db)

            if sample:
                # -------------------------------------------- Sampling --------------------------------------------------------------#
                if add_noise:
                    sampling_time, sampled_signal = sampling(sampling_frequency, Time, st.session_state.Summation)
                    st.session_state.fig.add_trace(
                        go.Scatter(visible=True, x=sampling_time, y=sampled_signal, mode='markers',
                                   name="Sampling With Noise")
                    )
                else:
                    sampling_time, sampled_signal = sampling(sampling_frequency, Time, st.session_state.Summation)
                    st.session_state.fig.add_trace(
                        go.Scatter(visible=True, x=sampling_time, y=sampled_signal, mode='markers', name="Sampling")
                    )
            if reconstruct:
                # ------------------------------------------ Reconstruction ---------------------------------------------------------------#

                reconstructed_signal, sam_time, sam_amp = sincInterpolation(sampling_frequency, Time,
                                                                            st.session_state.Summation)
                st.session_state.fig.add_trace(
                    go.Scatter(visible=True, x=Time, y=reconstructed_signal, name="Reconstruction")
                )

        plot()
        with content_middle:
            download_button = st.button('Download ️', key='download2')
            if download_button:
                download(Time, reconstructed_signal, st.session_state.max_added)


# -----------------------------------------Start and call main --------------------------------------------#

def start():
    if st.session_state.start == 0:
        interactivePlot()
        st.session_state.start = 1
    mainPage()


# -----------------------------------------Start the Website--------------------------------------------#
start()
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)