import pandas as pd
import numpy as np
import pickle

pickle_file = 'merged_data.pickle'
data_folder = 'C:/Research/IoT DDoS/'
files = ['DDoS-SlowLoris.pcap.csv',
         'DoS-HTTP_Flood.pcap.csv',
         'DoS-SYN_Flood.pcap.csv',
         'DDoS-SYN_Flood.pcap.csv',
         'DDoS-UDP_Fragmentation.pcap.csv',
         'DDoS-UDP_Flood.pcap.csv',
         'DDoS-SynonymousIP_Flood.pcap.csv',
         'DDoS-ICMP_Flood.pcap.csv',
         'DDoS-PSHACK_Flood.pcap.csv']

class_lbl_dict = {
    'DDoS-SlowLoris.pcap.csv': 0,
    'DoS-HTTP_Flood.pcap.csv': 1,
    'DoS-SYN_Flood.pcap.csv': 2,
    'DDoS-SYN_Flood.pcap.csv': 3,
    'DDoS-UDP_Fragmentation.pcap.csv': 4,
    'DDoS-UDP_Flood.pcap.csv' : 5,
    'DDoS-SynonymousIP_Flood.pcap.csv' : 6,
    'DDoS-ICMP_Flood.pcap.csv' : 7,
    'DDoS-PSHACK_Flood.pcap.csv' : 8
}

flag = False
merged_data = []
for file in files:
    df = pd.read_csv(data_folder + file)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna()

    data = df.values
    data = data[:, 0:-9]
    class_lbl_column = np.full((data.shape[0], 1), class_lbl_dict[file])
    print(class_lbl_column.shape, class_lbl_dict[file])
    data = np.hstack((data, class_lbl_column))
    print(data[:, -1].flatten())
    if flag == False:
        merged_data = data
        flag = True
    else:
        merged_data = np.vstack((merged_data, data))

    print(merged_data.shape)

fp = open(data_folder + pickle_file, 'wb')
pickle.dump(merged_data, fp)
fp.close()
