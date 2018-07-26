'''
класс Broker реализует API для связи с классом
FXStrategy
- экз бектеста
- экземпляр контекста биржи
'''
import v20
import datetime
import pandas as pd

class Broker:
    '''
    класс Broker реализует API для связи с классом
    FXStrategy
    '''
    def __init__(self):
        pass

    def get_pendingOrders(self,*args, **kwargs):
        pass

    def get_openTrades(self,*args, **kwargs):
        pass

    def create_order(self,*args, **kwargs):
        pass

    def close_trade(self, *args, **kwargs):
        pass

    def close_position(self,*args, **kwargs):
        pass

    def get_history(self, *args, **kwargs):
        pass

    def get_balance(self, *args, **kwargs):
        pass
    def get_commissions(self,*args, **kwargs):
        pass


class OANDA(Broker):

    def __init__(self,hostname='api-fxpractice.oanda.com',token=None,**kwargs):
        super(OANDA, self).__init__()
        # self.backtest = backtest.Backtest(self) циклическая ссылка!
        self.ctx = v20.Context(hostname, token=token,**kwargs)
        self.accountsList = self.ctx.account.list().body.get('accounts')
        self.accounts = [self.ctx.account.summary(account.id).body.get('account') for account in self.accountsList]

    def get_accounts(self):
        '''
        GET /v3/accounts and /v3/accounts/{accountID}/summary
        Аккаунты, связанные с токеном
        :return: [acc1,...]
        '''
        self.accountsList = self.ctx.account.list().body.get('accounts')
        self.accounts = [self.ctx.account.summary(account.id).body.get('account') for account in self.accountsList]
        return self.accounts

    def get_pendingOrders(self,*args, **kwargs):
        '''
        GET /v3/accounts/{accountID}/pendingOrders
        все висящие ордера
        :return: {'EUR':[{},{}],...}
        '''
        return {account.currency: self.ctx.order.list_pending(account.id).body.get('orders') for account in self.accounts}

    def get_openTrades(self):
        '''
        GET /v3/accounts/{accountID}/openTrades
        все открытые сделки
        :return: {'EUR': [],...}
        '''
        return {account.currency: self.ctx.trade.list_open(account.id).body.get('trades') for account in self.accounts}

    def create_order(self, currency, **specs):
        '''
        :param currency: валюта, связанная с аккаунтом
        :param specs:
        {
            #
            # Specification of the Order to create
            #
            order : (OrderRequest) (http://developer.oanda.com/rest-live-v20/order-df/#OrderRequest)
        }
        :return: response as json object (ujson lib)

        # example of creating EUR/USD Market Order to buy/sell 100 units:
            order = {
                "units": 100/-100,
                "instrument": "EUR_USD",
                "timeInForce": "FOK",
                "type": "MARKET",
                "positionFill": "DEFAULT"
            }
            broker.create_order('EUR', order=order)

        # example of creating a Take Profit Order @ 1.6000 for Trade with ID 6368
            order = {
                "timeInForce": "GTC",
                "price": 1.6000,
                "type": "TAKE_PROFIT",
                "tradeID": 6368
            }
            broker.create_order('EUR', order=order)

        # example of creating a Limit Order for -1000 USD_CAD @ 1.5000
        # with a Stop Loss on Fill @ 1.7000 and a Take Profit @ 1.14530
        order = {
            "price": 1.5000,
            "stopLossOnFill": {
              "timeInForce": "GTC",
              "price": 1.7000
            },
            "takeProfitOnFill": {
              "price": 1.14530
            },
            "timeInForce": "GTC",
            "instrument": "USD_CAD",
            "units": -1000,
            "type": "LIMIT",
            "positionFill": "DEFAULT"
        }
        '''
        for account in self.accounts:
            if currency == account.currency:
                return self.ctx.order.create(account.id, **specs)

    def close_trade(self,currency,tradeScpecifier=None, **specs):
        '''
        закрывает все открытые сделки по данной валюте
        PUT /v3/accounts/{accountID}/trades/{tradeSpecifier}/close
        :param currency: валюта, связанная с аккаунтом
        :param tradeScpecifier: tradeID
        :param specs:
        {
            #
            # Indication of how much of the Trade to close. Either the string “ALL”
            # (indicating that all of the Trade should be closed), or a DecimalNumber
            # representing the number of units of the open Trade to Close using a
            # TradeClose MarketOrder. The units specified must always be positive, and
            # the magnitude of the value cannot exceed the magnitude of the Trade’s
            # open units.
            #
            units : (string, default=ALL)
        }
        :return: [rsp_trade_id_#----,...]
        '''
        for account in self.accounts:
            if currency == account.currency:
                return [self.ctx.trade.close(account.id, trade.id, **specs) for trade in self.get_opentrades()[currency]]

    def close_position(self,currency,instrumentName,**specs):
        '''
        закрывает позиции по данной валюте (long или short) частично или полностью(см specs)
        :param currency: валюта, связанная с аккаунтом (account.currency)
        :param instrumentName: A string containing the base currency and quote currency delimited by a “_”.
        :param specs:
        {
            #
            # Indication of how much of the long Position to closeout. Either the
            # string “ALL”, the string “NONE”, or a DecimalNumber representing how many
            # units of the long position to close using a PositionCloseout MarketOrder.
            # The units specified must always be positive.
            #
            longUnits : (string, default=ALL),

            #
            # The client extensions to add to the MarketOrder used to close the long
            # position.
            #
            longClientExtensions : (ClientExtensions),

            #
            # Indication of how much of the short Position to closeout. Either the
            # string “ALL”, the string “NONE”, or a DecimalNumber representing how many
            # units of the short position to close using a PositionCloseout
            # MarketOrder. The units specified must always be positive.
            #
            shortUnits : (string, default=ALL),

            #
            # The client extensions to add to the MarketOrder used to close the short
            # position.
            #
            shortClientExtensions : (ClientExtensions)
        }
        :return:
        '''
        for account in self.accounts:
            if currency == account.currency:
                 self.ctx.position.close(account.id,instrumentName,**specs)

    def get_pricing(self,accountID=None,**specs):
        '''
        Get pricing information for a specified list of Instruments within an
        Account.
        :param: specs:
        {
            accountID:
                Account Identifier
            instruments:
                List of Instruments to get pricing for.(instruments='EUR_USD%2CUSD_CAD')
            since:
                Date/Time filter to apply to the response. Only prices and home
                conversions (if requested) with a time later than this filter
                (i.e. the price has changed after the since time) will be
                provided, and are filtered independently.
            includeUnitsAvailable:
                Flag that enables the inclusion of the unitsAvailable field in
                the returned Price objects.
            includeHomeConversions:
                Flag that enables the inclusion of the homeConversions field in
                the returned response. An entry will be returned for each
                currency in the set of all base and quote currencies present in
                the requested instruments list.
        }
        :return: v20.response.Response containing the results from submitting the
            request
        '''
        if not accountID:
            accountID = self.accountsList[0].id
        return self.ctx.pricing.get(accountID,**specs).body.get('prices')

    def get_history(self,instrumentName,**specs):
        '''
        GET /v3/instrument/{instrument}/candles

        :param instrumentName: A string containing the base currency and quote currency delimited by a “_”.
        :param specs:
            {
                'price': (string) The Price component(s) to get candlestick data for.
                 Can contain any combination of the characters “M” (midpoint candles)
                 “B” (bid candles) and “A” (ask candles). [default=M],

                'granularity': (http://developer.oanda.com/rest-live-v20/instrument-df/#CandlestickGranularity)
                 The granularity of the candlesticks to fetch [default=S5],

                'count': (integer)The number of candlesticks to return in the reponse.
                 Count should not be specified if both the start and end parameters are provided,
                 as the time range combined with the graularity will determine the number
                 of candlesticks to return. [default=500, maximum=5000],

                'from': (DateTime)The start of the time range to fetch candlesticks for,

                'to': (DateTime)The end of the time range to fetch candlesticks for,

                'smooth': (boolean)	A flag that controls whether the candlestick
                is “smoothed” or not. A smoothed candlestick uses the previous candle’s
                close price as its open price, while an unsmoothed candlestick uses
                the first price from its time range as its open price. [default=False],

                'includeFirst':	(boolean) A flag that controls whether the candlestick
                that is covered by the from time should be included in the results.
                This flag enables clients to use the timestamp of the last completed
                candlestick received to poll for future candlesticks but avoid receiving
                the previous candlestick repeatedly. [default=True],

                'dailyAlignment': (integer) The hour of the day (in the specified timezone)
                 to use for granularities that have daily alignments. [default=17, minimum=0, maximum=23],

                'alignmentTimezone': (string) The timezone to use for the dailyAlignment parameter.
                 Candlesticks with daily alignment will be aligned to the dailyAlignment hour within
                 the alignmentTimezone. Note that the returned times will still be represented in UTC.
                 [default=America/New_York],

                'weeklyAlignment': (WeeklyAlignment) The day of the week used for granularities that
                have weekly alignment. [default=Friday],

            }

            #example
                dt = datetime.datetime(2018, 7, 16, 0, 0, 0, 0)
                candles = broker.instrument_history('EUR_USD',price='BA',fromTime=broker.ctx.datetime_to_str(dt),granularity='M1',count=5000)
                for i in candles:
                    print(i,"\n")
        :return:
        '''
        return self.ctx.instrument.candles(instrumentName, **specs).body.get('candles')


if __name__ == '__main__':
    broker = OANDA(hostname='api-fxpractice.oanda.com',token="5fab1156c59dba001f91c7e329581e6d-fcec4321d69b2953c561bf7b511aface")
    order = {
        "units": 100,
        "instrument": "EUR_USD",
        "timeInForce": "FOK",
        "type": "MARKET",
        "positionFill": "DEFAULT"
    }
    """
    broker.create_order('EUR', order=order)
    print(broker.get_opentrades()['EUR'])
    dt = datetime.datetime(2018,7,16,0,0,0,0)
    prices=broker.get_pricing(instruments="EUR_USD", since=broker.ctx.datetime_to_str(dt))
    print(len(prices))
    """
    fromdt = datetime.datetime(2018, 1, 1, 0, 0, 0, 0)
    #todt=datetime.datetime(2018, 7, 13, 0, 0, 0, 0) toTime=broker.ctx.datetime_to_str(todt)
    #print(todt)
    candles = broker.get_history('EUR_USD',price='BA',fromTime=broker.ctx.datetime_to_str(fromdt),
                                        granularity='M10',count=5000)
    for i in candles:
        print(i,"\n")
    print(candles[-1].time)
    #broker.close_trade('EUR')

