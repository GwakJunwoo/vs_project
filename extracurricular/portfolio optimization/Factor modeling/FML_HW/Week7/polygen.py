import requests
import numpy as np
import pandas as pd

sector_matrix = []

s_list = np.load('D:\FBA Quant\_Assignment\FML\Week7\sector_t.npy')
cnt = 1

for s in s_list:
    symbols=s
    url = 'https://api.polygon.io/v1/meta/symbols/'+symbols+'/company?apiKey=TYqwde7tVnxB8BAb6cgUibXRpKxh6v8l'
    r = requests.get(url)
    try:
        sector_matrix.append(r.text.split(',')[9].split(':')[-1].split('"')[1])
    except:
        sector_matrix.append('None')
    print(cnt/9038*100)
    cnt += 1

