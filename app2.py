# ============================================================================
# Name        : Signal Studio (Task1)
# Author      : Team (24)
# Version     : last version
# Members     : Misara Ahmed-Rawan Abdelrahman-Rewan Mohmed-Rahma Abdelkader
# Supervision : Dr/Tamer - Eng/ Abdellah
# ============================================================================


# --------------------------------Imported Libraries--------------------------------#
from requests import session
from sqlalchemy import false
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import scipy
from scipy.signal import find_peaks

# --------------------------------Page Configuration--------------------------------#
st.set_page_config(layout="wide", page_icon='💡', page_title="Signal Studio")

# ----------------------------------Open css file------------------------------------#
with open("style.css") as css:
    st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)

# --------------------------------Session state for variables--------------------------------#
fig = go.Figure()
fig2 = go.Figure()

if 'fig' not in st.session_state or 'fig2' not in st.session_state or 'Signal' not in st.session_state or 'Summation' not in st.session_state:
    st.session_state.Signal = dict()
    st.session_state.fig = fig
    st.session_state.fig2 = fig2
    st.session_state.Summation = []

Time = np.linspace(0, 1, 500)
x = np.linspace(0, 1, 500)

# --------------------------------SideBar Options--------------------------------#
with st.sidebar:
    selected = option_menu(
        menu_title=None,
        options=['Home', 'Generate', 'Upload'],
        icons=['house', 'book'],
        menu_icon='cast',
        default_index=0,
    )


# ----------------------------------Home page-----------------------------------#

def home():
    st.header("Welcome to our Signal Studio 👋")
    st.markdown("Our website will give you multiple features to analyze your signals.")
    st.markdown(
        "As our website is designed for plotting your desired signals, add noise , sampling , reconstruction and also you can add or remove multiple signals.")
    st.markdown(
        "Our website provide saving each graph of your signal as png and zooming in or out, also you can save your signal in csv formate as well. ")


def sampling(sample_rate, signal):
    Time = np.linspace(0, 1, 500)
    T_sample = 1 / sample_rate
    n = np.arange(0, 1 / T_sample)
    sampling_time = n * T_sample
    sampled_signal = 0
    st.session_state.freq = []
    st.session_state.amp = []
    for i in st.session_state.Signal.keys():
        st.session_state.freq.append(st.session_state.Signal[i][0])
        st.session_state.amp.append(st.session_state.Signal[i][1])
    for i in st.session_state.Signal.keys():
        st.session_state.freq.append(st.session_state.Signal[i][0])
        st.session_state.amp.append(st.session_state.Signal[i][1])
    Amplitude_shift = scipy.signal.find_peaks(signal)
    t_axis = Time[Amplitude_shift[0]]
    sampling_time = sampling_time + t_axis[0]
    for frequency in range(len(st.session_state.freq)):
        sampled_signal += st.session_state.amp[frequency] * np.sin(
            2 * np.pi * st.session_state.freq[frequency] * sampling_time)
    return sampling_time, sampled_signal, T_sample


# Reconstructed Function
def sincInterpolation(sample_rate, signal):
    sampling_time, sampled_signal, T_sample = sampling(sample_rate, signal)
    sampled_signal = sampled_signal.reshape(sample_rate, 1)
    [nT, time] = np.meshgrid(sampling_time, Time, indexing='ij')
    y = np.sinc((time - nT) / T_sample) * sampled_signal
    reconstructed_signal = 0
    for i in range(sample_rate):
        reconstructed_signal += y[i, :]
    return reconstructed_signal

# *******************************************************************************************************
# Function Name: Generate
# Parameters (in): None
# Parameters (out): None
# Return value: None
# Description: Generation/removing of any desired signals with certain amplitudes and frequencies,
# adding noise to the combined signal also downloading the combined signal before and after adding
#  noise in CSV file.
# *********************************************************************************************************
def Generate():

    # st.session_state.Signal.clear()
    with st.form("my_form"):
        st.write("Add the signal")

        frequency = st.number_input("freq", min_value=0, max_value=100, value=0, step=1)
        amplitude = st.number_input("amp", min_value=0, max_value=100, value=0, step=1)
        submitted = st.form_submit_button("Add")

        if submitted:

            signal = amplitude * np.sin(frequency * 2 * np.pi * Time)

            st.session_state.Signal.update(
                {'Signal: ' + str(frequency) + ' hz, ' + str(amplitude) + ' cm': [frequency, amplitude, signal]})

            signal_Summation = 0

            for i in st.session_state.Signal.keys():
                signal_Summation += st.session_state.Signal[i][2]
                st.session_state.Summation = signal_Summation

                y = st.session_state.Signal[i][1] * np.sin(st.session_state.Signal[i][0] * 2 * np.pi * Time)
                fig2.add_trace(go.Scatter(visible=True, x=Time, y=y))
                st.session_state.fig2 = fig2

            fig.add_trace(go.Scatter(visible=True, x=Time, y=signal_Summation))
            st.session_state.fig = fig

    signal_To_Delete = st.selectbox(label="Select signal to delete:", options=st.session_state.Signal)

    def delete():
        if len(st.session_state.Signal) == 0:
            st.write("No Signal is added.")
            st.session_state.fig = go.Figure()

        else:
            st.session_state.Signal.pop(signal_To_Delete)
            signal_Summation = 0
            fig = go.Figure()
            fig2 = go.Figure()

            if len(st.session_state.Signal) > 0:
                for i in st.session_state.Signal.keys():
                    signal_Summation += st.session_state.Signal[i][2]
                    st.session_state.Summation = signal_Summation

                    y = st.session_state.Signal[i][1] * np.sin(st.session_state.Signal[i][0] * 2 * np.pi * Time)
                    fig2.add_trace(
                        go.Scatter(visible=True, x=Time, y=y)
                    )
                    st.session_state.fig2 = fig2

                fig.add_trace(go.Scatter(visible=True, x=Time, y=signal_Summation))
                st.session_state.fig = fig

            else:
                st.session_state.fig2 = go.Figure()
                st.session_state.fig = go.Figure()

    st.button("Delete ⛔", on_click=delete)
    st.subheader("Your signals separately")
    st.plotly_chart(st.session_state.fig2, use_container_width=True)

    st.subheader("Your combined signal ")
    st.plotly_chart(st.session_state.fig, use_container_width=True)

    # ---------------------------------------Download button--------------------------------------#

    download_combined = st.button('download your combined signal in .csv formate', key='download1')

    if download_combined:
        amplitudes = []
        frequencies = []
        signal_Summation = 0
        for i in st.session_state.Signal.keys():
            signal_Summation += st.session_state.Signal[i][2]
            amplitudes.append(st.session_state.Signal[i][1])
            frequencies.append(st.session_state.Signal[i][0])

        combined_Signal = {"X_axis": Time, "Y-axis": signal_Summation}
        combined_signal_two = {"Frequency": frequencies, "Amplitude": amplitudes}
        second_csv = pd.DataFrame(combined_signal_two)
        signal_csv = pd.DataFrame(combined_Signal)
        second_csv.to_csv("second file.csv", index=False)
        signal_csv.to_csv('Combined signal_final.csv', index=False)

    # --------------------------------Noise checkbox--------------------------------#
    noise_combined = st.checkbox('Add noise', key='noise_comb')

    if noise_combined:
        signal_Summation = 0

        for i in st.session_state.Signal.keys():
            signal_Summation += st.session_state.Signal[i][2]

        st.subheader("Control the noise in your signal through signal to noise ratio")
        target_snr_db = st.slider('Select a range for SNR', 0.0, 100.0, key='noise')
        st.write('SNR Values:', target_snr_db)

        if len(st.session_state.Summation) == 0:
            st.write("No Signal")
        else:
            signal_watts = (st.session_state.Summation) ** 2
            sig_avg_watts = np.mean(signal_watts)
            sig_avg_db = 10 * np.log10(sig_avg_watts)
            noise_avg_db = sig_avg_db - target_snr_db
            noise_avg_watts = 10 ** (noise_avg_db / 10)
            mean_noise = 0
            noise = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), len(signal_watts))
            noised_signal = st.session_state.Summation + noise
            noise_plot = px.line(x=Time, y=noised_signal)

            st.subheader("Your signal with noise")
            st.plotly_chart(noise_plot, use_container_width=True)

        # ---------------------------------------Download button--------------------------------------#

        download_combined_noise = st.button('download your noised signal in .csv formate', key='download4')
        if download_combined_noise:
            noised3_sig = {"X_axis": Time, "Y-axis": noised_signal}
            noised_csv = pd.DataFrame(noised3_sig)
            noised_csv.to_csv('Combined noised signals_final.csv', index=False)

    # ------------------------------------Sampling------------------------------------#
    if len(st.session_state.Signal) == 0:
        st.write("No Signal")
    else:
        # ------------------------------------Sampling and Reconstruction Apllaying------------------------------------#

        sampleRate = st.slider('Choose your sampling size', 1, 1000, key='key1')
        sampling_time, sampled_signal, T_sample = sampling(sampleRate, st.session_state.Summation)
        fig_s = go.Figure(data=go.Scatter(x=sampling_time, y=sampled_signal, mode='markers'))
        st.plotly_chart(fig_s, use_container_width=True)
        # ------------------------------------Linear interpolation------------------------------------#

        interpolated_fig = go.Figure(data=go.Scatter(x=sampling_time, y=sampled_signal, mode='lines+markers'))
        st.subheader("Your interpolated signal")
        st.plotly_chart(interpolated_fig, use_container_width=True)
        st.write(len(sampled_signal))
        reconstructed_signal = sincInterpolation(sampleRate, st.session_state.Summation)
        st.subheader("Your reconstructed signal")
        fig_res = px.line(x=Time, y=reconstructed_signal)
        st.plotly_chart(fig_res, use_container_width=True)
        # sampleRate = st.slider('Choose your sampling size', 1, 1000)

        # ---------------------------------------Download button--------------------------------------#

        download_comb_res = st.button('download your noised signal in .csv formate', key='download7')
        if download_comb_res:
            noised3_sig = {"X_axis": Time, "Y-axis": reconstructed_signal}
            noised_csv = pd.DataFrame(noised3_sig)
            noised_csv.to_csv('Combined reconstructed signals_final.csv', index=False)

        # --------------------------------Noise checkbox--------------------------------#

        noise_combined_res = st.checkbox('Add noise', key='noise_res')

        if noise_combined_res:
            signal_Summation = 0

            for i in st.session_state.Signal.keys():
                signal_Summation += st.session_state.Signal[i][2]

            st.subheader("Control the noise in your signal through signal to noise ratio")
            target_snr_db = st.slider('Select a range for SNR', 0.0, 100.0, key='noise_after_reconstruction')
            st.write('SNR Values:', target_snr_db)

            if len(st.session_state.Summation) == 0:
                st.write("No Signal")
            else:
                signal_watts = (st.session_state.Summation) ** 2
                sig_avg_watts = np.mean(signal_watts)
                sig_avg_db = 10 * np.log10(sig_avg_watts)
                noise_avg_db = sig_avg_db - target_snr_db
                noise_avg_watts = 10 ** (noise_avg_db / 10)
                mean_noise = 0
                noise = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), len(signal_watts))
                noised_signal = reconstructed_signal + noise

                noise_plot_res = px.line(x=Time, y=noised_signal)
                st.subheader("Your reconstructed signal with noise")
                st.plotly_chart(noise_plot_res, use_container_width=True)

                # ---------------------------------------Download button--------------------------------------#

                download_reconst_noise = st.button('download your noised signal in .csv formate', key='download2')
                if download_reconst_noise:
                    noised3_sig = {"X_axis": Time, "Y-axis": noised_signal}
                    noised_csv = pd.DataFrame(noised3_sig)
                    noised_csv.to_csv('Combined noised signals_final.csv', index=False)


# **********************************************************************************
# Function Name: Upload
# Parameters (in): None
# Parameters (out): None
# Return value: None
# Description: Plotting the uploaded CSV file.
# ***********************************************************************************
def Upload():
    st.session_state.Signal.clear()
    global fig
    upload_file = st.file_uploader('Upload your signal')
    if upload_file is not None:
        df = pd.read_csv(upload_file)
        x_axis_val = df.columns[0]
        y_axis_val = df.columns[1]
        # y_uploaded = df["Y-axis"].values
        frequency = df["Frequency"].values
        Amplitude = df["Amplitude"].values
        st.write(len(Amplitude))
        for object in range(len(Amplitude)):
            # st.write(object)
            y = Amplitude[object - 1] * np.sin(frequency[object - 1] * 2 * np.pi * Time)
            st.session_state.Signal.update(
                {'Signal: ' + str(frequency[object - 1]) + ' hz, ' + str(Amplitude[object - 1]) + ' cm': [
                    frequency[object - 1], Amplitude[object - 1], y]})
        st.write(st.session_state.Signal)
        add_Signal = st.checkbox("Add Signal(s)")
        add_Noise = st.checkbox("Add Noise")
        if add_Signal:

            with st.form("Adding_Signal_form"):
                st.write("Add the signal")

                frequency = st.number_input("freq", min_value=0, max_value=100, value=0, step=1)
                amplitude = st.number_input("amp", min_value=0, max_value=100, value=0, step=1)

                submitted = st.form_submit_button("Add")

                if submitted:
                    y = amplitude * np.sin(frequency * 2 * np.pi * Time)

                    st.session_state.Signal.update(
                        {'Signal: ' + str(frequency) + ' hz, ' + str(amplitude) + ' cm': [frequency, amplitude, y]})

            signal_To_Delete = st.selectbox(label="Select signal to delete:", options=st.session_state.Signal)

            def delete():
                if len(st.session_state.Signal) == 0:
                    st.write("No Signal is added.")
                    st.session_state.fig = go.Figure()


                else:
                    st.session_state.Signal.pop(signal_To_Delete)

            st.button("Delete ⛔", on_click=delete)

        if add_Noise:
            st.header("Control the noise in your signal through signal to noise ratio")
            target_snr_db = st.slider('Select a range for SNR', 0.0, 100.0)
            st.write('SNR Values:', target_snr_db)
            if len(st.session_state.Summation) == 0:
                st.write("No Signal")

            else:
                signal_watts = (st.session_state.Summation) ** 2
                sig_avg_watts = np.mean(signal_watts)
                sig_avg_db = 10 * np.log10(sig_avg_watts)
                noise_avg_db = sig_avg_db - target_snr_db
                noise_avg_watts = 10 ** (noise_avg_db / 10)
                mean_noise = 0
                noise = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), len(signal_watts))
                noised_signal = st.session_state.Summation + noise
                noised_figure = go.Figure()
                noised_figure.add_trace(
                    go.Scatter(visible=True, x=Time, y=noised_signal)
                )

                st.subheader("Noised Signal")
                st.plotly_chart(noised_figure, use_container_width=True)

                # ------------------------------------Sampling and Reconstruction Apllaying------------------------------------#

                sampleRate = st.slider('Choose your sampling size', 1, 1000, key='key1')
                sampling_time, sampled_signal, T_sample = sampling(sampleRate, st.session_state.Summation)
                fig_s = go.Figure(data=go.Scatter(x=sampling_time, y=sampled_signal, mode='markers'))
                st.plotly_chart(fig_s, use_container_width=True)
                # ------------------------------------Linear interpolation------------------------------------#

                interpolated_fig = go.Figure(data=go.Scatter(x=sampling_time, y=sampled_signal, mode='lines+markers'))
                st.subheader("Your interpolated signal")
                st.plotly_chart(interpolated_fig, use_container_width=True)
                st.write(len(sampled_signal))
                reconstructed_signal = sincInterpolation(sampleRate, st.session_state.Summation)
                st.subheader("Your reconstructed signal")
                fig_res = px.line(x=Time, y=reconstructed_signal)
                st.plotly_chart(fig_res, use_container_width=True)

        #  Reconstruction end
        signal_Summation = 0
        for i in st.session_state.Signal.keys():
            signal_Summation += st.session_state.Signal[i][2]
            st.session_state.Summation = signal_Summation
        fig.add_trace(
            go.Scatter(visible=True, x=Time, y=st.session_state.Summation)
        )
        st.session_state.fig = fig
        st.subheader("Original Signal")
        st.plotly_chart(st.session_state.fig, use_container_width=True)

    else:
        st.write("Upload your signal")
# def amplitudes_return(frame, signal):
#     list_of_amplitudes = []
#     n_samples = len(frame["Y-axis"].values)
#     np_fft = np.fft.fft(signal)
#     amplitudes = 2 / n_samples * np.abs(np_fft)
#     for objects in amplitudes:
#         if objects > 0.3:
#             if round(objects) not in list_of_amplitudes:
#                 list_of_amplitudes.append(round(objects))
#     # frequencies = np.fft.fftfreq(n_samples) * n_samples * 1 / (1 - 0)
#     return list_of_amplitudes
#
#
# def get_maxF():
#     if len(st.session_state.freq) == 0:
#         maxF = 1
#     else:
#         maxF = max(st.session_state.freq)
#     return maxF


# --------------------------------Sidebar pages --------------------------------#
if selected == 'Home':
    home()
elif selected == 'Generate':
    Generate()
elif selected == 'Upload':
    Upload()



