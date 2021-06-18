# COVID19 detection Research

Detection of COVID19 through voice using Neural Networks.

This work is part of a Masters Thesis submitted in partial fulfillment for the degree of Master of Science in Data Science for Worcester Polytechnic Institute

## Table of Contents

* [Dataset](#dataset)
* [Architectures](#architectures)
* [Setup](#setup)
* [Usage](#usage)
  * [Data generation](#data-generation)
  * [Training the network](#training-the-network)
  * [Inference](#inference)
* [Future work](#future-work-todo)
  * [Data Representations](#improve-on-data-representations)
  * [New Architecture try outs](#try-new-architectures)


## **Dataset**
We work on audio samples collected from [voca.ai](voca.ai) and [Coswara](https://github.com/iiscleap/Coswara-Data). Audio samples from both the datasets for combined and a 80-20 train-test stratified split is created. Below is the number of samples in each dataset.

|                                        | **Dataset**    | **Voca.ai**  | **Coswara**  |
| :------------------------------------: | :------------- | :------- | :----------- |
| **Cough Samples**                      | **Covid +ve**  | 1950     | 105          |
|                                        | **Covid -ve**  | 39       | 1361         |
| **Breath Samples**                     | **Covid +ve**  | -        | 103          |
|                                        | **Covid -ve**  | -        | 1366         |
| **Alphabet Samples**                   | **Covid +ve**  | 29       | -            |
|                                        | **Covid -ve**  | 1751     | -            |


## **Architectures**

Below are the architectures tried. All the files are under [networks](https://github.com/ShreeshaN/AlcoAudio/tree/master/alcoaudio/networks) folder. 


|Networks    | AUC  |
|---|---|
|  [Convolutional Neural Networks](https://github.com/ShreeshaN/COVID_19/blob/master/covid_19/runners/convnet_runner.py)(convnet) | 0.56  |
| [Conv Auto Encoders](https://github.com/ShreeshaN/COVID_19/blob/master/covid_19/runners/forced_autoencoder.py)(cae)  | 0.57  |
| [Variational Auto Encoders](https://github.com/ShreeshaN/COVID_19/blob/master/covid_19/runners/plain_conv_vae.py)(vae)  | 0.65  |
| [Contrastive Learning methods](https://github.com/ShreeshaN/COVID_19/blob/master/covid_19/runners/plain_conv_ae.py)(contrastive)  | 0.63  |
| [Brown et al.](https://arxiv.org/pdf/2006.05919.pdf)(Vggish + SVM)  | 0.61  |





## **Setup**

1. Download and run the requirements.txt to install all the dependencies.

      
       pip install -r requirements.txt
     
     
2. Create a [config](https://github.com/ShreeshaN/COVID_19/blob/master/covid_19/configs/model_configs_shree.json) file of your own


## Usage

### **Data generation**

Run ```data_processor.py``` to generate data required for training the model. It reads the raw audio samples, splits into ```n``` seconds and generates Mel filters, also called as Filter Banks (```fbank``` paramater in config file. Other available audio features are ```mfcc``` & ```gaf```)

    python3 covid_19/datagen/datadata_processor.py --config_file covid_19/configs/<config_filepath>

### **Training the network**

Using ```main.py``` one can train all the architectures mentioned in the above section.

    python3 main.py --config_file covid_19/configs/<config_filepath> --network convnet
        
### **Inference**

       
    python3 main.py --config_file --test_net True <config_filepath> --network convnet --datapath <data filepath>
       
   Remember to generate mel filters from raw audio data and use the generated ```.npy``` file for datapath parameter
   

## **Future work: TODO**

### **Improve on Data Representations**

 - [ ] Vocal Track Length Normalisation
 - [ ] Extract features using Praat and Opensmile
 - [ ] Normalise audio sample based on average amplitude

### **Try new architectures**

 - [ ] SincNet
 - [ ] [Graph Neural Networks](https://dl.acm.org/doi/10.1145/3450439.3451880)
