'''
модуль содержит различные ф-ии для
preprocessing данных
'''
from const import BROKERS
import pandas as pd
import os.path
import datetime


def candles_to_DataFrame(candles):
    '''
    Функция преобразует список ('историю') в pd.DataFrame
    (формат свечи соответствует брокеру OANDA)
    :param candles: [<v20.instrument.CandleStick>,...]
    :return:
    '''
    data = pd.DataFrame()
    for candle in candles:
        d = candle.dict()
        dt = datetime.datetime.strptime(d['time'],"%Y-%m-%dT%H:%M:%S.000000000Z")
        df = pd.DataFrame(data={'time': dt ,'volume': d['volume'],'complete': d['complete']},columns=['time','volume','complete'],index=[0])
        bid = pd.DataFrame(data=d['bid'],columns=['o', 'h', 'l', 'c'],index=[0],dtype='float64')
        ask = pd.DataFrame(data=d['ask'],columns=['o', 'h', 'l', 'c'],index=[0],dtype='float64')
        labels = [(l1, l2) for l1 in ['bid'] for l2 in ['o', 'h', 'l', 'c']]
        index = pd.MultiIndex.from_tuples(labels, names=['candle', 'prices'])
        bid.columns = pd.MultiIndex.from_tuples(index)
        labels = [(l1, l2) for l1 in ['ask'] for l2 in ['o', 'h', 'l', 'c']]
        index = pd.MultiIndex.from_tuples(labels, names=['candle', 'prices'])
        #index.rename('ask',level=0)
        ask.columns = pd.MultiIndex.from_tuples(index)
        df = pd.concat([df,bid,ask],axis=1)
        data = data.append(df,ignore_index=True)
    return data
def preprocess_data(data):
    xdata = data.values
    return xdata


