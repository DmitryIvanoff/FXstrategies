'''
модуль содержит различные ф-ии для
preprocessing данных
'''
from const import BROKERS
import pandas as pd
import os.path
from broker import OANDA


def mklbls(brokers,instruments,candle):
    labels = [(b, p, s) for b in brokers for p in instruments for s in candle]
    return labels

def load_data(files):
    '''
    Загружаем "историю"
    формируем шапку:
    broker    Bitfinex     ...  Kraken
    pair      BTC/USD ...       BTC/USD ...
    b/s       buy  sell         buy sell
    в переменной index и
    сливаем все df в один
    :param files:  from where we load data
    :return: returns pd.DataFrame
    '''
    index = pd.MultiIndex.from_tuples(mklbls(), names=['broker', 'pair', 'candle'])
    data = pd.DataFrame(columns=index)
    #print(data)
    for i in files:
        df = pd.read_csv(i)
        df.columns = index
        data = data.append(df, ignore_index=True)
        #print(data)
    return data

def dataset(data, brokers=None, pairs=None):
    '''
    формируем выборку из "истории"
    :param data:  pd.DataFrame, "история", с которой мы работаем
    :param brokers: список брокеров
    :param pairs: список валютных пар
    :return: returns ndarray
    '''
    if brokers is None:
        if pairs is None:
            array = data
        else:
            array = data.loc[:, (slice(None), pairs)]
    else:
        if pairs is None:
            array = data.loc[:, (brokers)]
        else:
            array = data.loc[:, (brokers, pairs)]
    array = array.as_matrix()
    return array

if __name__ == '__main__':
    #print(index)
    # print(nparr.shape)
    # print(nparr)
    data = load_data(map((lambda x: os.path.join(os.path.abspath('data'), x)), os.listdir('data')))

    #print (data)
    data = (dataset(data))
    print(data)
