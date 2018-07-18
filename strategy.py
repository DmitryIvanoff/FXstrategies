from const import VALUTE_PAIRS
from const import VALUTES
import numpy as np
import random, timeit, os
import data_processing as dp
import matplotlib as mpl
import matplotlib.pyplot as plt
from my_backtest import Backtest
from sklearn import metrics, preprocessing
from sklearn.externals import joblib
from models import LSTM_model
from models import MLP_model
np.random.seed(1335)  # for reproducibility
np.set_printoptions(precision=5, suppress=True, linewidth=150)


def init_state(data, time_step):
    """
    :param data: данные
    :return:
    """
    # diff = np.diff(close)
    # diff = np.insert(diff, 0, 0)
    # sma15 = SMA(data, timeperiod=15)
    # sma60 = SMA(data, timeperiod=60)
    # rsi = RSI(data, timeperiod=14)
    # atr = ATR(data, timeperiod=14)
    # --- Preprocess data
    # xdata = np.column_stack((close, diff, sma15, close - sma15, sma15 - sma60, rsi, atr))
    # xdata = np.re
    xdata = np.nan_to_num(data)
    scaler = preprocessing.StandardScaler()
    xdata = scaler.fit_transform(xdata)
    # joblib.dump(scaler, 'scaler.pkl')
    # scaler = joblib.load('scaler.pkl')
    state = xdata[time_step,0]
    # print(state)
    return state, xdata


# def get_state(data, time_step):
#     '''
#      состояние в момент времени time_step
#     :param time_step: time_step
#     :return:
#     '''
#     state = data[time_step, :]
#     return state

def save(name='', fmt='png'):
    # pwd = os.getcwd()
    # iPath = './pictures/{}'.format(fmt)
    # if not os.path.exists(iPath):
    #     os.mkdir(iPath)
    # os.chdir(iPath)
    plt.savefig('{}.{}'.format(name, fmt), fmt='png')
    # os.chdir(pwd)

def get_actions(state=None):
    '''
    возможные действия в данном состоянии
    по умолчанию все действия одинковы и содержатся в ACTIONS
    :param state:
    :return:
    '''
    actions = ACTIONS
    return actions




def take_action(current_state, data, action, time_step, broker):
    '''
    определяет следующее состояние
    и отображает принятое действие на
    :param current_state: текущее состояние
    :param data: данные
    :param action: текущее оптимальное действие
    :param time_step: момент времени
    :return: new_state, time_step, terminal_state
    '''
    # if the current iteration is the last state ("terminal state") then set terminal_state to 1
    if time_step + 1 == data.shape[0]-1:
        time_step+=1
        current_state = data[time_step, 0]
        terminal_st = 1
        # signal.loc[time_step] = 0
        # print(terminal_st)
        q = 1.0
        # if action == 1:
        #     if broker.balance['BTC'] < q:
        #         q = broker.balance['BTC']
        #     broker.balance['BTC'] -= 1.0 * q
        #     broker.balance['USD'] += broker.get_prices()['BTC/USD']['buy'] * q
        # elif action == -1:
        #     if broker.balance['USD'] < broker.get_prices()['BTC/USD']['sell'] * q:
        #         q = broker.balance['USD'] / broker.get_prices()['BTC/USD']['sell']
        #     broker.balance['BTC'] += 1.0 * q
        #     broker.balance['USD'] -= broker.get_prices()['BTC/USD']['sell'] * q
        return current_state, time_step, terminal_st, q
    time_step += 1
    broker.step()
    current_state = data[time_step, 0]
    #take action
    q=1.0
    # if action == 1:
    #      if broker.balance['BTC']<q:
    #         q=broker.balance['BTC']
    #      broker.balance['BTC']-=1.0*q
    #      broker.balance['USD']+=broker.get_prices()['BTC/USD']['buy']*q
    # elif action == -1:
    #      if broker.balance['USD']<broker.get_prices()['BTC/USD']['sell']*q:
    #          q=broker.balance['USD']/broker.get_prices()['BTC/USD']['sell']
    #      broker.balance['BTC']+=1.0*q
    #      broker.balance['USD']-=broker.get_prices()['BTC/USD']['sell']*q
         # broker.new_order()
    # print(current_state)
    terminal_st = 0
    # print(signal)
    return current_state, time_step, terminal_st, q


# def evaluate_Q(state, action, new_state, time_step):
#     Q_n = []
#     for a in ACTIONS:
#         q = model.predict(new_state, a, batch_size=1)
#         Q_n.append(q)
#     qval = get_reward(state, action, new_state, time_step) + max(Q_n)
#
#     return qval


# Get Reward, the reward is returned at the end of an episode
def get_reward(state, action, next_state, time_step, broker, q):
    reward = 0
    if action == -1:  #buy
        if q<=0.0:
            reward = 0
            pass
        else:
            reward = next_state-state
            # if reward <=0.0:
            #     reward=0
    elif action == 1: #sell
        if q<=0.0:
            reward = 0
            pass
        else:
            reward = state-next_state
            # if reward <=0.0:
            #     reward=0
    elif action == 0:
        reward = 0
    return reward

# ----------------------------------------------
# -------------main-----------------------------
ACTIONS = [-1, 0, 1] # buy(open) ignore sell(close)

def strategy(show=False,*params, **kwargs):
    # print(params)
    # TODO: 1. использовать данные через бектест напрямую
    # TODO: 2. принятие решений отображать на бектест
    # TODO: 3. reward вычислять , используя данные бектеста
    data = dp.load_data(map((lambda x: os.path.join(os.path.abspath('data'), x)), os.listdir('data')))
    data_all = dp.dataset(data)
    dataset_size = 1000
    data_bitfin = dp.dataset(data, ['Bitfinex'], ['BTC/USD'])[0:dataset_size]
    Broker_bitfin = Backtest('Bitfinex', data_all)
    Broker_bitfin.timestamp = 0#len(data_all)-1-dataset_size-3000
    input_dim = 2
    model = MLP_model(input_dim)
    start_time = timeit.default_timer()
    gamma = 0.95
    epsilon = 0.1
    batchSize = 100
    epochs = 1000
    storage_size = 1
    storage = []
    learning_progress = []
    h = 0
    # taken_actions = []
    # signals = [[], []]
    for i in range(epochs):
        time_step = 0
        taken_actions = []
        signals = [[],[]]#np.zeros((2, 0))
        state, xdata = init_state(data_bitfin, time_step)
        # print(state)
        # print(type(state))
        # print(xdata)
        status = 1
        Broker_bitfin.balance=Broker_bitfin.start_balance
        terminal_state = 0
        while status == 1:
            Q_n = []
            for a in get_actions(state):
                # qval = model.predict(np.concatenate((state.reshape(1, 1, 1), np.array([a]).reshape(1, 1, 1)), axis=2), batch_size=1)
                qval = model.predict(np.concatenate((np.array([state]).reshape(1,1), np.array([a]).reshape(1,1)), axis=1),batch_size=1)
                Q_n.append(qval)
            if (random.random() < epsilon):  # choose random action
                optimal_action = ACTIONS[np.random.randint(0, len(ACTIONS))]  # assumes 3 different actions
            else:  # choose best action from Q(s,a) values
                optimal_action = (np.argmax(Q_n)-1)
            # optimal_action = (np.argmax(Q_n) - 1)
            next_state, time_step, terminal_state, amount = take_action(state, xdata, optimal_action, time_step, Broker_bitfin)
            # if amount <= 0.0:
            #     optimal_action=0
            #print(time_step)
            reward = get_reward(state, optimal_action, next_state, time_step, Broker_bitfin, amount)
            if i == epochs-1:
                taken_actions.append(optimal_action)
            #np.append(signals[0], np.array([state]), axis=0)
                signals[0].append(state)
            # print(time_step,reward,optimal_action)
            # Experience replay storage
            if len(storage) < storage_size:  # if buffer not filled, add to it
                # storage.append((state.reshape(1, 1, 1), np.array([optimal_action]).reshape(1,1,1), reward, next_state.reshape(1, 1, 1)))
                storage.append((np.array([state]).reshape(1,1), np.array([optimal_action]).reshape(1,1), reward, np.array([next_state]).reshape(1,1)))
                # print(time_step, reward, terminal_state)
            else:  # if buffer full, overwrite old values
                if h < (storage_size - 1):
                    h += 1
                else:
                    h = 0
                # storage[h] = (state.reshape(1, 1, 1), np.array([optimal_action]).reshape(1, 1, 1), reward, next_state.reshape(1, 1, 1))
                storage[h] = (np.array([state]).reshape(1, 1), np.array([optimal_action]).reshape(1, 1), reward, np.array([next_state]).reshape(1,1))
                # randomly sample our experience replay memory
                #minibatch = random.sample(storage, batchSize)
                x_train = []
                y_train = []
                for memory in storage:
                    # Get max_Q(S',a)
                    old_state, action, reward, new_state = memory
                    # old_qval = model.predict(np.concatenate((old_state, action), axis=2), batch_size=1)
                    old_qval = model.predict(np.concatenate((old_state, action), axis=1), batch_size=1)
                    Q_n = []
                    # print(time_step,old_state,new_state)
                    for a in get_actions(new_state):
                        # qval = model.predict(np.concatenate((new_state, np.array([a]).reshape(1, 1, 1)), axis=2), batch_size=1)
                        qval = model.predict(np.concatenate((new_state, np.array([a]).reshape(1,1)), axis=1), batch_size=1)
                        Q_n.append(qval)
                    maxQ = np.max(Q_n)
                    newQ_n = (reward + (gamma * maxQ))
                    # print(old_qval, newQ_n)
                    # print(reward)
                    # x_train.append(np.concatenate((old_state, action), axis=2))
                    x_train.append(np.concatenate((old_state, action), axis=1))
                    y_train.append(newQ_n)
                x_train = np.squeeze(np.array(x_train), axis=1)
                y_train = np.array(y_train)
                model.fit(x_train, y_train, batch_size=storage_size, epochs=1, verbose=0)
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
        sc = plt.scatter(np.arange(len(taken_actions)), (np.array(signals[0])), c=taken_actions, cmap=cmap, vmin=-1,
                         vmax=1, marker='v')
        plt.colorbar(sc, label='Actions')
        plt.legend()
        save('exp2_' + str(i) + '_epoch', 'png')
        plt.show()



#from scipy.optimize import minimize

#simplex = None
#res = minimize(strategion, [ 0.88 ,  200  , -200 ],method='Nelder-Mead',options={'maxiter':1000,'disp':True,'initial_simplex':simplex})
#print(res.get('x'))
#strategy(res.get('x'),True)
strategy(show=True)
