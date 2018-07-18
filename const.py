#from debug import DEBUG


# Название базы данных InluxDB
DBNAME = 'valutes_prices'

# Дефолтный таймаут на обращение к API бирж
DEFAULT_TIMEOUT = 5  # seconds

# Префиксы для бирж
BROKERS_CRIPT = {
'EXMO':{
         'URLs': ['http://api.exmo.com/v1'],
         'currency':['BTC', 'LTC', 'USD', 'ETH', 'ZEC'],
         'INSTRUMENTS': [
            'BTC/USD', 'ETH/USD', 'ETH/BTC', 'LTC/USD',
              'LTC/BTC',  'ZEC/USD', 'ZEC/BTC',
          ]
},
'POLONIEX':{
         'URLs': ['https://poloniex.com/public','https://poloniex.com/tradingApi'],
         'currency':['BTC', 'LTC', 'USD', 'ETH', 'ZEC'],
        'INSTRUMENTS': [
            'BTC/USD', 'ETH/USD', 'ETH/BTC', 'LTC/USD',
            'LTC/BTC', 'ZEC/USD', 'ZEC/BTC',
        ]
},
'OKCOIN':{
    'URLs':['https://www.okcoin.com/api/v1'],
    'currency':['BTC', 'LTC', 'USD', 'ETH', 'ZEC'],
    'INSTRUMENTS': [
        'BTC/USD', 'ETH/USD', 'ETH/BTC', 'LTC/USD',
        'LTC/BTC', 'ZEC/USD', 'ZEC/BTC',
    ]
},
'BITFINEX':{
    'URLs':['https://api.bitfinex.com'],
    'currency':['BTC', 'LTC', 'USD', 'ETH', 'ZEC'],
    'INSTRUMENTS': [
        'BTC/USD', 'ETH/USD', 'ETH/BTC', 'LTC/USD',
        'LTC/BTC', 'ZEC/USD', 'ZEC/BTC',
    ]
},
}
BROKERS = {
    'OANDA':{
        'URLs':['api-fxpractice.oanda.com'],
        'currency':['EUR','USD'],
        'instruments': [
            'EUR_USD'
        ],
        'token':["5fab1156c59dba001f91c7e329581e6d-fcec4321d69b2953c561bf7b511aface"]
    },
}


# Коммиссии по переводам в процентах ( при переводе 0.08 эквивалента BTC)
WITHDRAW_FEE = {
    'BTC': 1.25,
    'ETH': 0.86,
    'LTC': 0.2,
    'ZEC': 0.07
}


# Пары валют, которые используются


# Если количество валюты (в пересчете на BTC) меньше этого предела,
# то не осуществлять перевод
VALUTE_BOUND_TO_WITHDRAW = 0.09  # BTC

# Процент от количества валюты для перевода
# (например, для кошелька с 1 ETH перевод 0.9 ETH)
SIZE_OF_WITHDRAW = 0.9  # percent

# Перекос в рублях для запуска алгоритма
# if DEBUG:
#     PROFIT_TO_EXECUTE = 0
# else:
#     PROFIT_TO_EXECUTE = 300
