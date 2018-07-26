from fxstrategy import FXStrategy
from broker import OANDA
import datetime
from const import BROKERS
import numpy as np
import random
import timeit, time
import os
import pandas as pd
import data_processing as dp
import matplotlib as mpl
import matplotlib.pyplot as plt
#from my_backtest import Backtest
from sklearn import metrics, preprocessing
from sklearn.externals import joblib
from models import LSTM_model
from models import MLP_model
import threading
import asyncio
np.random.seed(1335)  # for reproducibility
np.set_printoptions(precision=5, suppress=True, linewidth=150)


class QStrategy(FXStrategy):
    '''
    подкласс класса FXStrategy
    QStrategy реализует торговлю на бирже
    с помощью Q-обучения
    '''
    def __init__(self):
        super(QStrategy, self).__init__()
        self.broker = OANDA(hostname=BROKERS['OANDA']['URLs'][0],
                            token=BROKERS['OANDA']['token'][0])
                            #datetime_format="UNIX")
        self.actions = [
            (0,0,0), #no action
            (1,0,0), #long(buy) open
            (1,1,0), #short(sell) open
            (1,0,1), #long(buy) close
            (1,1,1)  #short(sell) close
        ]
        self.history = []
        #self.broker.ctx.set_datetime_format_unix()

    def get_state(self, data, time_step):
        """
        :param data: обработанные данные np.ndarray(len(history), len(state))
        :return: np.ndarray
        """
        state = data[time_step, :]
        #print(state)
        return state

    def load_history(self,
                     instrumentName,
                     fromTime,
                     granularity='M10',
                     count=5000,
                     **kwargs):
        # todt=datetime.datetime(2018, 7, 13, 0, 0, 0, 0) toTime=broker.ctx.datetime_to_str(todt)
        # print(todt)
        async def lol():
            candles = self.broker.get_history(instrumentName,
                                              price='BA',
                                              fromTime=self.broker.ctx.datetime_to_str(fromTime),
                                              granularity=granularity,
                                              count=count,
                                              includeFirst=False,
                                              **kwargs)
            await asyncio.sleep(0.002)
            while (candles and candles[-1].complete):
                for candle in candles:
                    self.history.append(candle)
                candles = self.broker.get_history(instrumentName,
                                                  price='BA',
                                                  fromTime=candles[-1].time,
                                                  granularity=granularity,
                                                  count=count,
                                                  includeFirst=False,
                                                  **kwargs)
                await asyncio.sleep(0.002)
            if candles:
                for candle in candles:
                    self.history.append(candle)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(lol())
        loop.close()

    def get_curr_state(self, data, time_step):
        '''
         состояние в момент времени time_step
        :param time_step: time_step
        :return:
        '''
        state = data[time_step, :]
        return state

    def save(self, name='', fmt='png'):
        # pwd = os.getcwd()
        # iPath = './pictures/{}'.format(fmt)
        # if not os.path.exists(iPath):
        #     os.mkdir(iPath)
        # os.chdir(iPath)
        plt.savefig('{}.{}'.format(name, fmt), fmt='png')
        # os.chdir(pwd)

    def get_actions(self, state=None):
        '''
        возможные действия в данном состоянии
        по умолчанию все действия одинковы и содержатся в ACTIONS
        :param state:
        :return:
        '''
        return self.actions

    def take_action(self, current_state, data, action, time_step, broker):
        pass

    def get_reward(self, state, action, next_state, data , time_step, broker=None):
        reward = 0
        if (action[0]==1):
            if (action[1]==1):
                #short
                if (action[2] == 1):
                    #close
                    reward = 0#-(data[time_step,6]+data[time_step,5])/2
                else:
                    #open
                    reward = ((data[time_step,1]+data[time_step,2])/2)-((data[time_step+1,6]+data[time_step+1,5])/2)#((data[time_step,1]+data[time_step,2])/2)
            else:
                #long
                if (action[2] == 1):
                    #close
                    reward = 0#(data[time_step,1]+data[time_step,2])/2
                else:
                    #open
                    reward = (data[time_step+1,1]+data[time_step+1,2])/2-(data[time_step,6]+data[time_step,5])/2#-(data[time_step,6]+data[time_step,5])/2
        else:
            pass

        return reward

    def save_experience(self,storage,storage_size,time_step,state,opt_action,next_state,gamma,data):
        reward = self.get_reward(state, opt_action, next_state,data,time_step, self.broker)
        h = len(storage)-1
        newQ_n=0.0
        if len(storage) < storage_size:  # if buffer not filled, add to it
            # storage.append((state.reshape(1, 1, 1), np.array([optimal_action]).reshape(1,1,1),
            #  reward, next_state.reshape(1, 1, 1)))
            storage.append((state.reshape(1, state.shape[-1]), opt_action.reshape(1, opt_action.shape[-1]), reward,
                           next_state.reshape(1, next_state.shape[-1])))
        else:
            if h < (storage_size-1):
                h += 1
            else:
                h = 0
            # storage[h] = (state.reshape(1, 1, 1), np.array([optimal_action]).reshape(1, 1, 1),
            # reward, next_state.reshape(1, 1, 1))
            storage[h] = (state.reshape(1, state.shape[-1]), opt_action.reshape(1, opt_action.shape[-1]), reward,
                           next_state.reshape(1, next_state.shape[-1]))
            x_train = []
            y_train = []
            #print(storage)
            for memory in storage:
                old_state, action, reward, new_state = memory
                # old_qval = model.predict(np.concatenate((old_state, action), axis=2), batch_size=1)
                old_qval = self.model.predict(np.concatenate((old_state, action), axis=1), batch_size=1)
                Q_n = []
                # print(time_step,old_state,new_state)
                for a in self.get_actions(new_state):
                    # qval = model.predict(np.concatenate((new_state, np.array([a]).reshape(1, 1, 1)), axis=2),
                    #  batch_size=1)
                    a = np.array(a)
                    #print(new_state)
                    qval = self.model.predict(np.concatenate((new_state, a.reshape(1, a.shape[-1])), axis=1),
                                              batch_size=1)
                    Q_n.append(qval)
                maxQ = np.max(Q_n)
                newQ_n = (reward + (gamma * maxQ))
                #print(old_qval, newQ_n, reward)
                # print(reward)
                # x_train.append(np.concatenate((old_state, action), axis=2))
                x_train.append(np.concatenate((old_state, action), axis=1))
                y_train.append(newQ_n)
            x_train = np.squeeze(np.array(x_train), axis=1)
            y_train = np.array(y_train)
            self.model.fit(x_train, y_train, batch_size=storage_size, epochs=1, verbose=0)
        return newQ_n

    def get_curr_state(self,data,time_step=None):
        pass

    def get_opt_action(self,state, epsilon):
        Q_n=[]
        actions=self.get_actions(state)
        for action in actions:
            a = np.array(action)
            # qval = model.predict(np.concatenate((state.reshape(1, 1, state.shape[-1]), a.reshape(1, 1, a.shape[-1])), axis=2), batch_size=1)
            qval = self.model.predict(
                np.concatenate((state.reshape(1, state.shape[-1]), a.reshape(1, a.shape[-1])), axis=1),
                batch_size=1)
            #print(np.concatenate((state.reshape(1, state.shape[-1]), a.reshape(1, a.shape[-1])), axis=1))
            Q_n.append(qval)
        if (random.random() < epsilon):  # choose random action
            optimal_action = actions[np.random.randint(0, len(actions))]
        else:                            # choose best action from Q(s,a) values
            optimal_action = actions[np.argmax(Q_n)]
        return np.array(optimal_action)

    def trade(self,interval='M10',*args, **kwargs):
        '''

        :param interval:
        :param args:
        :param kwargs:
        :return:
        '''
        epochs = kwargs.get('epochs')
        if not epochs:
            epochs = 1000
        self.load_history('EUR_USD', datetime.datetime(2018, 1, 1, 0, 0, 0, 0))
        (prev_opt_action,prev_state) = self.train(epochs,interval)
        while True:
            state = self.get_curr_state()
            actions = self.get_actions(state)
            opt_action = self.get_optimal(state,actions)
            err = self.take_action(state,opt_action)
            if not err:
                self.save_experience(prev_state,prev_opt_action,state)
            prev_state = state
            prev_opt_action = opt_action
            time.sleep(600)

    def train(self,epochs=100,interval='M10',*args, **kwargs):
        '''

        :param epochs:
        :param interval:
        :param kwargs:
        :return: (prev_opt_action, prev_state)
        '''
        show = kwargs.get('show')
        data = dp.candles_to_DataFrame(self.history)
        # print(data)
        raw_data = data.loc[:, slice(('bid', 'o'), ('ask', 'c'))].values
        # diff = np.diff(close)
        # diff = np.insert(diff, 0, 0)
        # sma15 = SMA(data, timeperiod=15)
        # sma60 = SMA(data, timeperiod=60)
        # rsi = RSI(data, timeperiod=14)
        # atr = ATR(data, timeperiod=14)
        # --- Preprocess data
        # data = np.column_stack((close, diff, sma15, close - sma15, sma15 - sma60, rsi, atr))
        scaler = preprocessing.StandardScaler()
        # joblib.dump(scaler, 'scaler.pkl')
        # scaler = joblib.load('scaler.pkl')
        data = scaler.fit_transform(raw_data)
        print(data)
        dataset_size = data.shape[0]#1000
        input_dim = data.shape[1]+len(self.actions[0])
        print(input_dim)
        self.model = MLP_model(input_dim)
        start_time = timeit.default_timer()
        gamma = 0.95
        epsilon = 0.1
        batchSize = 100
        storage_size = 1
        storage = []
        h = 0
        # taken_actions = []
        # signals = [[], []]
        for i in range(epochs):
            time_step = 0 if (data.shape[0]-dataset_size)<=0 else data.shape[0]-dataset_size
            taken_actions = []
            signals = [[], []]  # np.zeros((2, 0))
            state = self.get_state(data,time_step)
            status = 1
            #Broker_bitfin.balance = Broker_bitfin.start_balance
            terminal_state = 0
            while status == 1:
                opt_action = self.get_opt_action(state,epsilon)
                # take_action()
                # broker.step()
                if time_step + 1 == data.shape[0] - 1:
                    terminal_state = 1
                else:
                    terminal_state = 0
                time_step += 1
                next_state = self.get_state(data, time_step)
                newQ_n = self.save_experience(storage,storage_size, time_step-1,state,opt_action,next_state,gamma,raw_data)
                state = next_state
                if terminal_state == 1:  # if reached terminal state, update epoch status
                    status = 0
                # eval_reward = evaluate_Q(test_data, model, price_data, i)
                #  learning_progress.append((eval_reward))
            print("Epoch #: %d Reward: %f Epsilon: %f" % (i, newQ_n, epsilon))
            # learning_progress.append(reward)
            if epsilon > 0.00:  # decrement epsilon over time
                epsilon -= (1.0 / epochs)
        if show:
            plt.title('Experiment №2.' + str(i) + '_epoch')
            plt.ylabel('state')
            plt.xlabel('time steps')
            plt.plot(signals[0], color='yellow', label='sell price')
            # plt.plot(signals[1], color='green', label='buy price')
            cmap = mpl.cm.get_cmap('RdBu', 3)
            plt.grid(True)
            sc = plt.scatter(np.arange(len(taken_actions)), (np.array(signals[0])),
                             c=taken_actions,
                             cmap=cmap, vmin=-1,
                             vmax=1, marker='v')
            plt.colorbar(sc, label='Actions')
            plt.legend()
            self.save('exp2_' + str(i) + '_epoch', 'png')
            plt.show()

if __name__ == '__main__':
    strategy = QStrategy()
    strategy.load_history('EUR_USD', datetime.datetime(2018, 6, 1, 0, 0, 0, 0))
    (prev_opt_action, prev_state) = strategy.train()
    '''
    state = self.get_curr_state()
    actions = self.get_actions(state)
    opt_action = self.get_optimal(state, actions)
    err = self.take_action(state, opt_action)
    if not err:
    self.save_experience(prev_state, prev_opt_action, state)
    prev_state = state
    prev_opt_action = opt_action
    time.sleep(600)
    '''