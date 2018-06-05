#from debug import DEBUG


# Название базы данных InluxDB
DBNAME = 'valutes_prices'

# Дефолтный таймаут на обращение к API бирж
DEFAULT_TIMEOUT = 5  # seconds

# Префиксы для бирж
EXMO = 'exmo'
POLONIEX = 'poloniex'
OKCOIN = 'okcoin'
BITFINEX = 'bitfinex'

# URL для доступа к Exmo API
EXMO_API_URL = 'http://api.exmo.com/v1'
# URL для доступа к Bitfinex
BITFINEX_URL = 'https://api.bitfinex.com'
# URL для доступа к Poloniex API
POLONIEX_PUBLIC_URL = 'https://poloniex.com/public'
POLONIEX_TRADING_URL = 'https://poloniex.com/tradingApi'
# URL для доступа к OKCoin API
OKCOIN_API_URL = 'https://www.okcoin.com/api/v1'

# Типы ордеров
BUY = 'buy'
SELL = 'sell'

# Используемые валюты
VALUTES = ['BTC', 'LTC', 'USD', 'ETH', 'ZEC']

# Коммиссии по переводам в процентах ( при переводе 0.08 эквивалента BTC)
WITHDRAW_FEE = {
    'BTC': 1.25,
    'ETH': 0.86,
    'LTC': 0.2,
    'ZEC': 0.07
}

# Сокращения названий валют и их расшифровки
VALUTE_FULL_NAMES = {
    'BTC': 'bitcoin',
    'ETH': 'ethereum',
    'LTC': 'litecoin',
    'ZEC': 'zcash',
    'USD': 'usd'
}

# Пары валют, которые используются
VALUTE_PAIRS = [
    'BTC/USD', 'ETH/USD', 'ETH/BTC', 'LTC/USD',
    'LTC/BTC',  'ZEC/USD', 'ZEC/BTC',
    ]

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
