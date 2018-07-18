from const import BROKERS


class Backtest:
    '''

    '''
    balance = {
              'BTC': 3,
              'USD': 20000,
              'ETH': 1,
              'LTC': 0,
              'ZEC': 0
              }
    start_balance = {
                    'BTC': 3,
                    'USD': 20000,
                    'ETH': 1,
                    'LTC': 0,
                    'ZEC': 0
                    }
    oborot = 0

    def __init__(self, broker, data):
        self.data = data
        self.broker = broker
        self.history_size = len(data)
        self.money_arr = []
        self.stock_money_arr = []
        self.orders = []
        self.prices_all = []
        self.timestamp = 0
        self.prices_all = self.data_to_prices()
        self.old_prices = 100 * [0]
    def data_to_prices(self):
        '''

        :return:список словарей [timestamp, {pair: {buy: , sell:}}]
        '''
        out_price = []
        if self.broker == 'Bitfinex':
            sdvig = 0
        if self.broker == 'Poloniex':
            sdvig = 14
        if self.broker == 'Exmo':
            sdvig = 28
        if self.broker == 'Gdax':
            sdvig = 42
        if self.broker == 'Kraken':
            sdvig = 56
        for j in range(0, len(self.data)):
            price_arr = []
            for price in self.data[j]:
                try:
                    price_arr.append(price)
                except ValueError:
                    price_arr.append(0)
            out = {}
            for i in range(0, 7, 1):
                buy = price_arr[sdvig + 2 * i]
                sell = price_arr[sdvig + 2 * i + 1]
                save_new = True
                if buy == 0:
                    buy = self.old_prices[sdvig + 2 * i]
                    save_new = False
                if sell == 0:
                    sell = self.old_prices[sdvig + 2 * i + 1]
                    save_new = False
                out.update({VALUTE_PAIRS[i]: {'buy': buy, 'sell': sell}})
            if save_new:
                self.old_prices = price_arr
            out_price.append(out)
        return out_price



    @staticmethod
    def get_money_in_usd(prices, balance):
        '''
        all our money in USD (denominator=USD)
        :param prices: (dict)
        :param balance: (dict)
        :return: (float)
        '''
        money = 0
        for i in VALUTES:
            if i == 'USD':
                money += balance[i]
            else:
                money += balance[i] * prices[i + '/USD']['sell']
        return money



    def step(self):
        '''

        :return:
        '''
        if self.timestamp >= self.history_size - 1:
            pass
        else:
            self.timestamp += 1
            prices = self.get_prices()
            for idx, ord in enumerate(self.orders):
                valutes_pair = ord['valutes_pair']
                price = ord['price']
                amount = ord['amount']
                order_type = ord['type']
                market_price = prices[valutes_pair][order_type]
                if (order_type == 'buy') and (market_price <= price):
                    self.orders.pop(idx)
                    # self.oborot += self.prices[valutes_pair.split('/')[0] + '/USD'][order_type] * amount
                    # Увеличить первую валюту
                    self.balance[valutes_pair.split('/')[0]] += amount * 0.998  # 0.2% комиссия
                    # Уменьшить вторую валюту
                    #self.balance[valutes_pair.split('/')[1]] -= prices[valutes_pair][order_type] * amount

                if (order_type == 'sell') and (market_price >= price):
                    self.orders.pop(idx)
                    # self.oborot += self.prices[valutes_pair.split('/')[0] + '/USD'][order_type] * amount
                    # Уменьшить первую валюту
                    #self.balance[valutes_pair.split('/')[0]] -= amount
                    # Увеличить вторую валюту
                    self.balance[valutes_pair.split('/')[1]] += amount*market_price* 0.998  # 0.2% комиссия
            self.money_arr.append(self.get_money_in_usd(prices, self.balance))
            self.stock_money_arr.append(self.get_money_in_usd(prices, self.start_balance))

    def new_order(self, valutes_pair, price, amount, order_type, now=False):
        if now:
            market_price = self.prices[valutes_pair][order_type]
            if (order_type == 'buy'):
                # Увеличить первую валюту
                self.oborot += self.prices[valutes_pair.split('/')[0] + '/USD'][order_type] * amount
                self.balance[valutes_pair.split('/')[0]] += amount * 0.998  # 0.2% комиссия
                # Уменьшить вторую валюту
                self.balance[valutes_pair.split('/')[1]] -= amount * market_price

            if (order_type == 'sell'):
                # Уменьшить первую валюту
                self.oborot += self.prices[valutes_pair.split('/')[0] + '/USD'][order_type] * amount
                self.balance[valutes_pair.split('/')[0]] -= amount
                # Увкличить вторую валюту
                self.balance[valutes_pair.split('/')[1]] += amount * market_price * 0.998  # 0.2% комиссия
        else:
            if (order_type == 'buy'):
                self.balance[valutes_pair.split('/')[1]] -= amount * price
            if (order_type == 'sell'):
                self.balance[valutes_pair.split('/')[0]] -= amount
            self.orders.append(dict(
                valutes_pair=valutes_pair,
                price=price,
                amount=amount,
                type=order_type,
            ))

    def withdraw(self, address, val, amount):
        fee = {'BTC': 0.0005,
               'ETH': 0.01,
               'LTC': 0.001,
               'USD': 25,
               'ZEC': 0.001}
        self.balance[val] -= amount
        address.balance[val] += amount - fee[val]

    def get_prices(self):
        """
        цены на текущий момент
        :return:
        """
        return self.prices_all[self.timestamp]

    def return_balance_history(self):
        return self.money_arr, self.stock_money_arr

    def orders_cancel(self):
        for order in self.orders:
            if order['type'] == 'sell':
                self.balance[order['valutes_pair'][0:3]] += order['amount']
            else:
                self.balance[order['valutes_pair'][-3:]] += order['amount'] * order['price']
        self.orders = []

