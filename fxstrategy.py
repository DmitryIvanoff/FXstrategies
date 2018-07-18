from broker import Broker
from abc import abstractmethod


class FXStrategy:
    '''
    класс FXStrategy реализует
    базовый класс стратегии для торговли на бирже
    содержит:
    - брокер (открытие/закрытие сделок, выставление ордеров, выбор валют и т.д.)
    - *метод run (запускающий стратегию в отдельном потоке(процессе))
    -
    '''
    def __init__(self):
        pass
    @abstractmethod
    def trade(self):
        pass

    def run(self):
        '''
        запускает стратегию в отдельном потоке/процессе
        :return:
        '''
        self.trade(self)
        pass


