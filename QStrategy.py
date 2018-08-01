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
            (0,0,0), #no action       0
            (1,0,0), #long(buy) open  4
            (1,1,0), #short(sell) open 6
            (1,0,1), #long(buy) close  5
            (1,1,1)  #short(sell) close 7
        ]
        self.history = []
        self.balance = 50
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
                    if self.short_pos:
                        for r in self.short_pos:
                            reward+=(r - state[1])
                        self.short_pos.clear()
                        self.balance+=reward
                    else:
                        reward = 0
                else:
                    #open
                    if self.balance>0:
                        reward = state[0]-next_state[1]#self.balance
                        self.short_pos.append(state[0])
                    else:
                        reward = 0
                    self.balance = (self.balance - state[0]) if (self.balance - state[0])>0 else 0
            else:
                #long
                if (action[2] == 1):
                    #close
                    if self.long_pos:
                        for r in self.long_pos:
                            reward+=(state[0]-r)
                        self.long_pos.clear()
                        self.balance += reward
                    else:
                        reward = 0
                else:
                    #open
                    if (self.balance>0):
                        reward = next_state[0]-state[1]#self.balance
                        self.long_pos.append(state[1])
                    else:
                        reward = 0
                    self.balance = self.balance - state[1] if (self.balance - state[1]) > 0 else 0
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
                #old_qval = self.model.predict(np.concatenate((old_state, action), axis=1), batch_size=1)
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
                #print(opt_action,reward)
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
        actions=self.get_actions(state)
        Q_n=np.zeros(len(actions))
        for k in range(len(actions)):
            a = np.array(actions[k])
            # qval = model.predict(np.concatenate((state.reshape(1, 1, state.shape[-1]), a.reshape(1, 1, a.shape[-1])), axis=2), batch_size=1)
            qval = self.model.predict(
                np.concatenate((state.reshape(1, state.shape[-1]), a.reshape(1, a.shape[-1])), axis=1),
                batch_size=1)
            #print(np.concatenate((state.reshape(1, state.shape[-1]), a.reshape(1, a.shape[-1])), axis=1))
            Q_n[k]=qval
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
        df = dp.candles_to_DataFrame(self.history)
        # print(data)
        raw_data = df.loc[:, (('bid', 'o'),('ask', 'o'),('bid', 'c'),('ask', 'c'))].values
        # diff = np.diff(close)
        # diff = np.insert(diff, 0, 0)
        # sma15 = SMA(data, timeperiod=15)
        # sma60 = SMA(data, timeperiod=60)
        # rsi = RSI(data, timeperiod=14)
        # atr = ATR(data, timeperiod=14)
        # --- Preprocess data
        # data = np.column_stack((close, diff, sma15, close - sma15, sma15 - sma60, rsi, atr))
        #scaler = preprocessing.StandardScaler()
        # joblib.dump(scaler, 'scaler.pkl')
        # scaler = joblib.load('scaler.pkl')
        raw_data = raw_data-1.0#scaler.fit_transform(raw_data)
        bid_change = (raw_data[:,2]-raw_data[:,0])*100.0/raw_data[:,0]
        bid_change = bid_change.reshape(bid_change.shape[0],1)
        ask_change = (raw_data[:,3]-raw_data[:,1])*100.0/raw_data[:,1]
        ask_change = ask_change.reshape(ask_change.shape[0], 1)
        #print(bid_change)
        #print(raw_data[:, (2, 3)])
        data = np.concatenate((raw_data[:,(2,3)],bid_change,ask_change),axis=1)
        print(data)
        #print(df)
        dataset_size = data.shape[0]#1000
        input_dim = data.shape[1]+len(self.actions[0])
        self.model = MLP_model(input_dim)
        start_time = timeit.default_timer()
        gamma = 0.95
        epsilon = 1
        multiplier=10
        batchSize = 100
        storage_size = 1
        storage = []
        h = 0
        # taken_actions = []
        # signals = [[], []]
        balance = 20
        for i in range(epochs):
            self.balance = balance
            self.long_pos = []
            self.short_pos = []
            maxQ = np.NINF
            #print (maxQ)
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
                if show:
                    signals[0].append(state[0])
                    signals[1].append(state[1])
                    a=0
                    for k in range(len(opt_action)):
                        a+=opt_action[k]*(2**(len(opt_action)-k-1))
                    taken_actions.append(a)
                if time_step + 1 == data.shape[0] - 1:
                    terminal_state = 1
                else:
                    terminal_state = 0
                time_step += 1
                next_state = self.get_state(data, time_step)
                newQ = self.save_experience(storage,storage_size, time_step-1,state,opt_action,next_state,gamma,raw_data)
                maxQ = newQ if newQ>maxQ else maxQ
                state = next_state
                if terminal_state == 1:  # if reached terminal state, update epoch status
                    status = 0
                # eval_reward = evaluate_Q(test_data, model, price_data, i)
                #  learning_progress.append((eval_reward))
            print("Epoch #: %d Max_Q: %f Epsilon: %f Balance: %f" % (i, maxQ, epsilon,self.balance))
            # learning_progress.append(reward)
            if epsilon > 0.00:  # decrement epsilon over time
                epsilon -= (multiplier*1.0 / epochs)
            if show:
                plt.title('Experiment №3. Epoch №'+str(i))
                plt.ylabel('state')
                plt.xlabel('time steps')
                plt.plot(signals[0], color='black', label='bid price',linewidth=0.5)
                # plt.plot(signals[1], color='green', label='ask price')
                #cmap = mpl.cm.get_cmap('RdBu', 3)
                plt.grid(True)
                sc = plt.scatter(np.arange(len(taken_actions)),(np.array(signals[0])),
                                 c=taken_actions,
                                 cmap=plt.cm.get_cmap('Accent'),
                                 vmin=min(taken_actions),
                                 vmax=max(taken_actions),
                                 marker='*')
                plt.colorbar(sc, label='Actions')
                plt.legend()
                if not (i%100):
                    self.save('exp3_' + str(i) + '_epoch', 'png')
                    plt.show()
                if not (i%200) and i:
                    epsilon=1
                    multiplier=50
                plt.clf()
        return opt_action

if __name__ == '__main__':
    strategy = QStrategy()
    strategy.load_history('EUR_USD', datetime.datetime(2018, 7, 29, 0, 0, 0, 0))
    (prev_state) = strategy.train(epochs=1000,show=True)
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