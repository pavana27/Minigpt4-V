import pandas as pd
import numpy as np
from scipy.signal import butter, lfilter, sosfilt
import matplotlib.pyplot as plt
import os

def transpose_csv(input_file_path, output_file_path):
    # Read the CSV file
    df = pd.read_csv(input_file_path)

    # Transpose the DataFrame
    df_transposed = df.transpose()

    # Write the transposed DataFrame to a new CSV file without header
    df_transposed.to_csv(output_file_path, index=False, header=False)

    
def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos

def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    #y = lfilter(sos, 1, data)
    y = sosfilt(sos, data)
    return y

def extract_breathing_signal(rppg_signal, sampling_rate, lowcut=0.15, highcut=0.4):
    breathing_signal = butter_bandpass_filter(rppg_signal, lowcut, highcut, sampling_rate)
    return breathing_signal

if __name__ == '__main__':
    """
    #original rppg csv file has rppg siglans ecxtarcted and saved row wise 
    #it is easier to convert rows to columns and process by columns(eacg rppg signals as a dataframe)
    input_file_path = '/Users/pavana/Desktop/crema.csv'
    output_file_path = '/Users/pavana/Desktop/crema-transposed.csv'

    #transpose_csv(input_file_path, output_file_path)
    #print(f"Transposed data saved to {output_file_path}")
    
    #now read the transposed file and extarct rppg sigbnals by columns
    #save extracted breath signals columns wise into new csv file
    #same goes with hr signals change the low and high cut frequency and the output file path
    #input_file_path = '/Users/pavana/Desktop/crema.csv'
    """
    output_file_path = '/Users/pavana/Desktop/sed_4_rppg.csv'
    
    # Read the input CSV file
    df = pd.read_csv(output_file_path)
    sampling_rate = 30  # should be same for both HR and BR
    # Print the number of columns in the input CSV file
    num_columns = df.shape[1]
    print(f"Number of columns in the input CSV file: {num_columns}")
    #Number of columns in the input CSV file: 9003
    extracted_signals = pd.DataFrame()

    extracted_file_path = '/Users/pavana/Desktop/sed_4.csv'
    # Process each column in the DataFrame
    for column in df.columns[:4]:
        rppg_signal = df[column]
        breathing_signal = extract_breathing_signal(rppg_signal, sampling_rate)
        extracted_signals[column] = breathing_signal

    # Save the extracted breathing signals to a new CSV file without header
    extracted_signals.to_csv(extracted_file_path, index=False, header=False)
    
    
    #print(f"Heart Rate signals extracted and saved to {output_file_path}")
    # Read the saved breathing signals CSV file
    extracted_df = pd.read_csv(extracted_file_path, header=None)

    # Plot the first five breathing signals from the first five columns
    plt.figure(figsize=(10, 6))
    #labels = ['Angry', 'Disappointed', 'Fear', 'Nuetral', 'SAD']
    labels = ['Barely Engaged', 'Engaged', 'Highly Engaged', 'Not Engaged']
    for i in range(min(4, extracted_df.shape[1])):
        plt.plot(extracted_df.iloc[:, i], label=labels[i],linewidth=2)
    #plt.plot(extracted_df, label='HR Signal')
    plt.xlabel('Time (samples)',fontsize=16)
    plt.ylabel('Amplitude',fontsize=16)
    plt.title('BR Signals',fontsize=16)
    plt.legend()
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.show()
    