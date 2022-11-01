<center>  <h1> Signal Studio</h1 >
</center>

# Table of contents
* [Information](#Information)
* [Features](#Features)
* [Task Info](#TaskInfo)
* [Screenshots](#Screenshots)


<hr>

# Information
- Our website is made by streamlit -python-. <br>
- All graphs are implemeted using plotly. <br>

<hr>

# Features
> General Information
- Our website will give you multiple features to analyze your signals:
  > As our website is designed for :
  - plotting your desired signals by determine signals frequency and amplitude.
  - Add Multiple signals at the same time and show the final output signal.
  - Upload signal from your files in csv formateand show it.
  - Remove any signal you have added by selecting and deleting it.
  - Add noise to your signal by determine signal to noise ratio(SNR) Which has a range if SNR=0 that means noise is extrem and if SNR=50 that means signal will appear without noise.
  - sample your signal with 2 options:
  > By determine frequency you want in HZ OR
  > By sampling with Max_frequency Scale (I wnant to sample with fmax,2fmax.....,10fmax)
  - reconstruct your signal and this has more one option (you can't reconstrict it nicely if your sampling frecency <2fmax (according to nquist's law ), and you can reconstruct it well if sampling frequency >= 2fmax)
  - Our website provide saving each graph of your signal as png and zooming in or out, also you can doenload your reconstructed  signal in csv formate as well.
<hr>

# Task Info
> **Course**: Digital Signal Processing

>**3rd Year, 1st Semester**

> **Date**: 1/11/2022

>**Team**:
  - Misara Ahmed, Sec.2, B.N. 43
  - Rawan mohamed, Sec.1  B.N. 34
  - Rawan Abdelrahman, Sec.1, B.N. 32
  - Rahma Abdelkader, Sec.1  B.N. 31
  

<hr>


# Screeshots
-These following screenshots will show our features .<br>
## Our website with default signal before any action
![Screan](https://user-images.githubusercontent.com/93431157/199153214-560d3088-0158-4a74-a9e9-2647c81086e2.png)
## Adding 2 signal with different frequency
![Adding_signals](https://user-images.githubusercontent.com/93431157/199153732-37d80034-e940-45cb-b520-fe54a3e1ad09.png)
-There are many scenarios when sampling: 
## Sampling with frequency scale with sampling frequency < 2 fmax (Fmax=6.5 Hz as shown )
![under_sampling](https://user-images.githubusercontent.com/93431157/199154684-4441b896-68fb-403a-8d1e-34fa6dce4731.png)
## Sampling with frequency scale with sampling frequency = 2 fmax (Fmax=6.5 Hz as shown )
![nequist](https://user-images.githubusercontent.com/93431157/199156003-d029bbdf-b5ef-4514-aeb9-276f64970311.png)
## Samplind with frequency scale with sampling frequency > 2 fmax (Fmax=6.5 Hz as shown )
![oversampling](https://user-images.githubusercontent.com/93431157/199156365-7835ac73-f9eb-424b-b89b-40611b562735.png)
## Sampling 2 signals (4hz&8hz) with sampling frequency=6hz
![case png](https://user-images.githubusercontent.com/93431157/199159514-7154bd41-f222-4be2-a6fb-1289f89fb6f1.jpeg)
## sampling with 2fmax with noise 
![noise](https://user-images.githubusercontent.com/93431157/199157783-2f9bedae-b787-4de1-8389-1e7495af4352.png)
## sampling with Fmax scale 
![Fmax](https://user-images.githubusercontent.com/93431157/199158193-60469411-d9c4-4cd3-9ab8-f240703ccca4.png)
## Uploading & downloading 
![Download](https://user-images.githubusercontent.com/93431157/199158492-7f2ebf17-1a5f-4c0e-8762-c4dca444f635.png)
![Uploading](https://user-images.githubusercontent.com/93431157/199158548-67dbc12b-0b98-4a79-a337-4b2f11a33ba8.png)

<hr>
