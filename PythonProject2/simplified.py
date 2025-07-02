import numpy as np
import pandas as pd
import asyncio
import logging
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from enum import Enum
import requests
import json
from datetime import datetime, timedelta
import hashlib
import hmac
import time
import os

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Enums для типов данных
class OrderType(Enum):
    BUY = "BUY"
    SELL = "SELL"


class TradingSignal(Enum):
    HOLD = 0
    BUY = 1
    SELL = 2


@dataclass
class Position:
    symbol: str
    size: float
    entry_price: float
    stop_loss: float
    take_profit: float
    timestamp: datetime


@dataclass
class MarketData:
    symbol: str
    price: float
    volume: float
    timestamp: datetime
    high: float
    low: float
    open_price: float
    close_price: float


# Простая нейронная сеть без PyTorch
class SimpleNeuralNetwork:
    def __init__(self, input_size: int = 20):
        self.input_size = input_size
        # Случайные веса для демонстрации
        np.random.seed(42)
        self.weights_1 = np.random.randn(input_size, 10) * 0.1
        self.weights_2 = np.random.randn(10, 3) * 0.1
        self.bias_1 = np.zeros((10,))
        self.bias_2 = np.zeros((3,))

        # Параметры для нормализации
        self.feature_means = None
        self.feature_stds = None

    def sigmoid(self, x):
        """Сигмоидная функция активации"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def softmax(self, x):
        """Softmax для финального слоя"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def normalize_features(self, features):
        """Нормализация входных признаков"""
        if self.feature_means is None:
            self.feature_means = np.mean(features, axis=0) if len(features.shape) > 1 else features
            self.feature_stds = np.std(features, axis=0) + 1e-8 if len(features.shape) > 1 else np.ones_like(features)

        return (features - self.feature_means) / self.feature_stds

    def predict(self, features: np.ndarray) -> tuple:
        """Предсказание торгового сигнала"""
        try:
            # Нормализация входных данных
            if len(features) != self.input_size:
                # Приведение к нужному размеру
                if len(features) > self.input_size:
                    features = features[:self.input_size]
                else:
                    features = np.pad(features, (0, self.input_size - len(features)), 'constant')

            features = self.normalize_features(features)

            # Прямое распространение
            hidden = self.sigmoid(np.dot(features, self.weights_1) + self.bias_1)
            output = self.softmax(np.dot(hidden, self.weights_2) + self.bias_2)

            # Получение предсказания и уверенности
            prediction = np.argmax(output)
            confidence = np.max(output)

            return prediction, confidence

        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            return 0, 0.5  # HOLD с низкой уверенностью

    def train_simple(self, features_list: List[np.ndarray], labels: List[int], epochs: int = 50):
        """Простое обучение методом градиентного спуска"""
        learning_rate = 0.01

        for epoch in range(epochs):
            total_loss = 0
            for features, label in zip(features_list, labels):
                # Приведение к нужному размеру
                if len(features) != self.input_size:
                    if len(features) > self.input_size:
                        features = features[:self.input_size]
                    else:
                        features = np.pad(features, (0, self.input_size - len(features)), 'constant')

                features = self.normalize_features(features)

                # Прямое распространение
                hidden = self.sigmoid(np.dot(features, self.weights_1) + self.bias_1)
                output = self.softmax(np.dot(hidden, self.weights_2) + self.bias_2)

                # Вычисление ошибки
                target = np.zeros(3)
                target[label] = 1
                loss = -np.sum(target * np.log(output + 1e-8))
                total_loss += loss

                # Простое обновление весов (упрощенная версия)
                output_error = output - target
                self.weights_2 -= learning_rate * np.outer(hidden, output_error)
                self.bias_2 -= learning_rate * output_error

            if epoch % 10 == 0:
                avg_loss = total_loss / len(features_list)
                logger.info(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")


class SimpleAPIClient:
    """Упрощенный API клиент для демонстрации"""

    def __init__(self, base_url: str = "", api_key: str = "", secret_key: str = ""):
        self.base_url = base_url
        self.api_key = api_key
        self.secret_key = secret_key
        self.demo_mode = True  # Демо режим для тестирования

    def get_demo_price(self, symbol: str) -> float:
        """Генерация демо цены для тестирования"""
        base_prices = {
            'BTCUSDT': 45000,
            'ETHUSDT': 3000,
            'ADAUSDT': 0.5
        }
        base_price = base_prices.get(symbol, 100)
        # Добавляем случайные колебания ±2%
        variation = np.random.uniform(-0.02, 0.02)
        return base_price * (1 + variation)

    def make_request(self, endpoint: str, method: str = "GET", params: Dict = None) -> Dict:
        """Выполнение запроса (демо версия)"""
        if self.demo_mode:
            # Возвращаем демо данные
            if 'klines' in endpoint:
                symbol = params.get('symbol', 'BTCUSDT')
                base_price = self.get_demo_price(symbol)

                # Генерируем демо свечи
                candles = []
                for i in range(params.get('limit', 100)):
                    timestamp = int(time.time() * 1000) - (i * 60000)  # Минутные свечи
                    price_variation = np.random.uniform(-0.01, 0.01)
                    price = base_price * (1 + price_variation)

                    candles.append([
                        timestamp,  # Время открытия
                        str(price * 0.999),  # Цена открытия
                        str(price * 1.001),  # Максимальная цена
                        str(price * 0.998),  # Минимальная цена
                        str(price),  # Цена закрытия
                        str(np.random.uniform(100, 1000)),  # Объем
                    ])

                return candles[::-1]  # Возвращаем в хронологическом порядке

            elif 'order' in endpoint:
                return {
                    'symbol': params.get('symbol'),
                    'orderId': np.random.randint(1000000, 9999999),
                    'status': 'FILLED',
                    'executedQty': params.get('quantity'),
                    'price': self.get_demo_price(params.get('symbol', 'BTCUSDT'))
                }

        return {}


class MarketDataHandler:
    def __init__(self, api_client: SimpleAPIClient, symbols: List[str], interval: str = "1m"):
        self.api_client = api_client
        self.symbols = symbols
        self.interval = interval
        self.data_cache = {}
        self.last_update = datetime.now()

    def fetch_candles(self, symbol: str, limit: int = 100) -> List[Dict]:
        """Получение свечных данных"""
        params = {
            'symbol': symbol,
            'interval': self.interval,
            'limit': limit
        }

        data = self.api_client.make_request('/api/v3/klines', params=params)

        candles = []
        for candle in data:
            candles.append({
                'timestamp': datetime.fromtimestamp(int(candle[0]) / 1000),
                'open': float(candle[1]),
                'high': float(candle[2]),
                'low': float(candle[3]),
                'close': float(candle[4]),
                'volume': float(candle[5])
            })

        return candles

    def update(self):
        """Обновление рыночных данных"""
        for symbol in self.symbols:
            try:
                candles = self.fetch_candles(symbol)
                self.data_cache[symbol] = candles
                logger.info(f"Market data updated for {symbol}: {len(candles)} candles")
            except Exception as e:
                logger.error(f"Failed to update data for {symbol}: {e}")

        self.last_update = datetime.now()

    def calculate_technical_indicators(self, candles: List[Dict]) -> Dict:
        """Расчет технических индикаторов"""
        if len(candles) < 20:
            return {}

        closes = np.array([c['close'] for c in candles])
        volumes = np.array([c['volume'] for c in candles])

        # Скользящие средние
        sma_5 = np.mean(closes[-5:])
        sma_10 = np.mean(closes[-10:])
        sma_20 = np.mean(closes[-20:])

        # RSI (упрощенная версия)
        price_changes = np.diff(closes[-15:])
        gains = price_changes[price_changes > 0]
        losses = np.abs(price_changes[price_changes < 0])

        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0

        rsi = 100 - (100 / (1 + (avg_gain / (avg_loss + 1e-8))))

        # MACD (упрощенная версия)
        ema_12 = np.mean(closes[-12:])
        ema_26 = np.mean(closes[-26:]) if len(closes) >= 26 else np.mean(closes)
        macd = ema_12 - ema_26

        return {
            'sma_5': sma_5,
            'sma_10': sma_10,
            'sma_20': sma_20,
            'rsi': rsi,
            'macd': macd,
            'current_price': closes[-1],
            'volume': volumes[-1]
        }

    def get_features(self, symbol: str) -> np.ndarray:
        """Извлечение признаков для ML модели"""
        if symbol not in self.data_cache or len(self.data_cache[symbol]) < 20:
            return np.array([])

        candles = self.data_cache[symbol]
        indicators = self.calculate_technical_indicators(candles)

        if not indicators:
            return np.array([])

        # Создание вектора признаков
        features = np.array([
            indicators['current_price'],
            indicators['sma_5'],
            indicators['sma_10'],
            indicators['sma_20'],
            indicators['rsi'],
            indicators['macd'],
            indicators['volume'],
            # Добавляем последние 13 цен для достижения размера 20
            *[c['close'] for c in candles[-13:]]
        ])

        return features


class PositionManager:
    def __init__(self, max_risk_per_trade: float = 0.02):
        self.open_positions: List[Position] = []
        self.max_risk_per_trade = max_risk_per_trade

    def is_open(self, symbol: str) -> bool:
        return any(pos.symbol == symbol for pos in self.open_positions)

    def add_position(self, position: Position):
        self.open_positions.append(position)
        logger.info(f"Position added: {position.symbol} - {position.size}")

    def remove_position(self, symbol: str):
        self.open_positions = [pos for pos in self.open_positions if pos.symbol != symbol]
        logger.info(f"Position closed: {symbol}")

    def get_position_size(self, account_balance: float, entry_price: float, stop_loss: float) -> float:
        risk_amount = account_balance * self.max_risk_per_trade
        price_difference = abs(entry_price - stop_loss)
        if price_difference > 0:
            return min(risk_amount / price_difference, account_balance * 0.1)  # Максимум 10% от баланса
        return 0.0


class SimpleLogger:
    """Простое логирование в файл"""

    def __init__(self, log_file: str = "transactions.log"):
        self.log_file = log_file
        self.write_log("=== Trading System Started ===")

    def write_log(self, message: str):
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                timestamp = datetime.now().isoformat()
                f.write(f"{timestamp} | {message}\n")
        except Exception as e:
            print(f"Failed to write log: {e}")

    def log_transaction(self, user_id: str, transaction_type: str, amount: float,
                        symbol: str = None, status: str = "COMPLETED"):
        message = f"TRANSACTION | {user_id} | {transaction_type} | {amount} | {symbol} | {status}"
        self.write_log(message)
        logger.info(f"Transaction logged: {transaction_type} {amount} for user {user_id}")


class UserAccount:
    def __init__(self, user_id: str, initial_balance: float = 10000.0):
        self.user_id = user_id
        self.balance = initial_balance
        self.transaction_logger = SimpleLogger()

    def get_balance(self) -> float:
        return self.balance

    def update_balance(self, amount: float, transaction_type: str = "TRADE"):
        old_balance = self.balance
        self.balance += amount

        self.transaction_logger.log_transaction(
            self.user_id, transaction_type, amount
        )

        logger.info(
            f"Balance updated for user {self.user_id}: ${old_balance:.2f} -> ${self.balance:.2f} ({amount:+.2f})")


class TradeExecutor:
    def __init__(self, api_client: SimpleAPIClient, position_manager: PositionManager, user_account: UserAccount):
        self.api_client = api_client
        self.position_manager = position_manager
        self.user_account = user_account

    def place_order(self, symbol: str, order_type: OrderType, quantity: float, price: float = None) -> bool:
        """Размещение ордера"""
        try:
            # Проверка баланса
            if order_type == OrderType.BUY:
                required_balance = quantity * (price or self.api_client.get_demo_price(symbol))
                if required_balance > self.user_account.balance:
                    logger.warning(f"Insufficient balance: {self.user_account.balance} < {required_balance}")
                    return False

            params = {
                'symbol': symbol,
                'side': order_type.value,
                'type': 'MARKET' if price is None else 'LIMIT',
                'quantity': quantity,
            }

            if price:
                params['price'] = price

            result = self.api_client.make_request('/api/v3/order', method='POST', params=params)

            if result.get('status') == 'FILLED':
                # Обновление баланса
                executed_price = float(result.get('price', price or self.api_client.get_demo_price(symbol)))
                if order_type == OrderType.BUY:
                    self.user_account.update_balance(-quantity * executed_price, "BUY")
                else:
                    self.user_account.update_balance(quantity * executed_price, "SELL")

                logger.info(f"Order executed: {symbol} {order_type.value} {quantity} @ {executed_price}")
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            return False

    def close_position(self, symbol: str) -> bool:
        """Закрытие позиции"""
        try:
            position = next((pos for pos in self.position_manager.open_positions
                             if pos.symbol == symbol), None)

            if not position:
                logger.warning(f"No open position found for {symbol}")
                return False

            # Размещение ордера на закрытие
            opposite_side = OrderType.SELL if position.size > 0 else OrderType.BUY
            success = self.place_order(symbol, opposite_side, abs(position.size))

            if success:
                # Расчет P&L
                current_price = self.api_client.get_demo_price(symbol)
                pnl = position.size * (current_price - position.entry_price)

                self.position_manager.remove_position(symbol)
                logger.info(f"Position closed: {symbol}, P&L: ${pnl:.2f}")

            return success

        except Exception as e:
            logger.error(f"Failed to close position: {e}")
            return False


class SimpleTradingSystem:
    def __init__(self, user_id: str, symbols: List[str], initial_balance: float = 10000.0):
        self.user_account = UserAccount(user_id, initial_balance)
        self.api_client = SimpleAPIClient()  # Демо режим
        self.market_data = MarketDataHandler(self.api_client, symbols)
        self.neural_network = SimpleNeuralNetwork()
        self.position_manager = PositionManager()
        self.trade_executor = TradeExecutor(self.api_client, self.position_manager, self.user_account)

        self.is_running = False
        self.symbols = symbols

        # Инициализация "обученной" модели с случайными весами
        self._initialize_model()

    def _initialize_model(self):
        """Инициализация модели случайными данными для демонстрации"""
        logger.info("Initializing neural network with demo data...")

        # Генерируем случайные обучающие данные
        np.random.seed(42)
        training_features = []
        training_labels = []

        for _ in range(100):
            features = np.random.randn(20)  # Случайные признаки
            # Простая логика для создания меток: если средние признаки > 0, то BUY
            if np.mean(features[:5]) > 0.1:
                label = TradingSignal.BUY.value
            elif np.mean(features[:5]) < -0.1:
                label = TradingSignal.SELL.value
            else:
                label = TradingSignal.HOLD.value

            training_features.append(features)
            training_labels.append(label)

        # "Обучение" модели
        self.neural_network.train_simple(training_features, training_labels, epochs=20)
        logger.info("Neural network initialized")

    async def start(self):
        """Запуск торговой системы"""
        logger.info("Starting Simple Trading System...")
        logger.info(f"Initial balance: ${self.user_account.balance:.2f}")
        self.is_running = True

        # Начальное обновление данных
        self.market_data.update()

        # Запуск торгового цикла
        await self._trading_loop()

    async def stop(self):
        """Остановка системы"""
        logger.info("Stopping trading system...")
        self.is_running = False

        # Закрытие всех позиций
        for position in self.position_manager.open_positions[:]:
            self.trade_executor.close_position(position.symbol)

        logger.info(f"Final balance: ${self.user_account.balance:.2f}")
        logger.info("Trading system stopped")

    async def _trading_loop(self):
        """Основной торговый цикл"""
        iteration = 0
        max_iterations = 20  # Ограничение для демонстрации

        while self.is_running and iteration < max_iterations:
            try:
                iteration += 1
                logger.info(f"=== Trading Iteration {iteration} ===")

                # Обновление рыночных данных
                self.market_data.update()

                # Обработка каждого символа
                for symbol in self.symbols:
                    await self._process_symbol(symbol)

                # Проверка стоп-лоссов
                self._check_stop_losses()

                # Вывод статистики
                self._print_status()

                # Пауза между итерациями
                await asyncio.sleep(5)  # 5 секунд для демонстрации

            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(2)

        # Автоматическая остановка после демонстрации
        await self.stop()

    async def _process_symbol(self, symbol: str):
        """Обработка торгового сигнала для символа"""
        try:
            # Получение признаков
            features = self.market_data.get_features(symbol)
            if len(features) == 0:
                return

            # Предсказание нейронной сети
            prediction, confidence = self.neural_network.predict(features)
            signal_name = ['HOLD', 'BUY', 'SELL'][prediction]

            current_price = features[0]  # Текущая цена - первый признак

            logger.info(f"{symbol}: {signal_name} (confidence: {confidence:.3f}, price: ${current_price:.2f})")

            # Выполнение торговых действий
            if confidence > 0.6:  # Порог уверенности
                await self._execute_trade(symbol, prediction, confidence, current_price)

        except Exception as e:
            logger.error(f"Error processing symbol {symbol}: {e}")

    async def _execute_trade(self, symbol: str, prediction: int, confidence: float, current_price: float):
        """Выполнение торговой операции"""
        try:
            is_position_open = self.position_manager.is_open(symbol)

            if prediction == TradingSignal.BUY.value and not is_position_open:
                # Покупка
                stop_loss_price = current_price * 0.98  # 2% стоп-лосс
                position_size = self.position_manager.get_position_size(
                    self.user_account.balance, current_price, stop_loss_price
                )

                if position_size > 0.001:  # Минимальный размер позиции
                    success = self.trade_executor.place_order(symbol, OrderType.BUY, position_size)

                    if success:
                        position = Position(
                            symbol=symbol,
                            size=position_size,
                            entry_price=current_price,
                            stop_loss=stop_loss_price,
                            take_profit=current_price * 1.06,  # 6% тейк-профит
                            timestamp=datetime.now()
                        )
                        self.position_manager.add_position(position)

            elif prediction == TradingSignal.SELL.value and is_position_open:
                # Продажа (закрытие позиции)
                success = self.trade_executor.close_position(symbol)

        except Exception as e:
            logger.error(f"Error executing trade for {symbol}: {e}")

    def _check_stop_losses(self):
        """Проверка стоп-лоссов"""
        for position in self.position_manager.open_positions[:]:
            if position.symbol in self.market_data.data_cache:
                current_price = self.api_client.get_demo_price(position.symbol)

                # Проверка стоп-лосса
                if ((position.size > 0 and current_price <= position.stop_loss) or
                        (position.size < 0 and current_price >= position.stop_loss)):
                    logger.warning(f"Stop loss triggered for {position.symbol}: {current_price}")
                    self.trade_executor.close_position(position.symbol)

                # Проверка тейк-профита
                elif ((position.size > 0 and current_price >= position.take_profit) or
                      (position.size < 0 and current_price <= position.take_profit)):
                    logger.info(f"Take profit triggered for {position.symbol}: {current_price}")
                    self.trade_executor.close_position(position.symbol)

    def _print_status(self):
        """Вывод текущего статуса системы"""
        balance = self.user_account.balance
        open_positions = len(self.position_manager.open_positions)

        logger.info(f"Balance: ${balance:.2f}, Open Positions: {open_positions}")

        for pos in self.position_manager.open_positions:
            current_price = self.api_client.get_demo_price(pos.symbol)
            pnl = pos.size * (current_price - pos.entry_price)
            logger.info(f"  {pos.symbol}: Size={pos.size:.4f}, Entry=${pos.entry_price:.2f}, "
                        f"Current=${current_price:.2f}, P&L=${pnl:.2f}")


# Главная функция для запуска
async def main():
    """Главная функция для демонстрации системы"""
    print("🚀 Запуск упрощенной торговой системы...")
    print("📊 Система работает в ДЕМО режиме с виртуальными данными")
    print("💰 Начальный баланс: $10,000")
    print("-" * 50)

    # Конфигурация
    USER_ID = "demo_user"
    SYMBOLS = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
    INITIAL_BALANCE = 10000.0

    # Создание и запуск системы
    trading_system = SimpleTradingSystem(USER_ID, SYMBOLS, INITIAL_BALANCE)

    try:
        await trading_system.start()
    except KeyboardInterrupt:
        logger.info("Получен сигнал прерывания...")
        await trading_system.stop()
    except Exception as e:
        logger.error(f"Ошибка в главной функции: {e}")
        await trading_system.stop()


# Запуск системы
if __name__ == "__main__":
    # Проверка доступности модулей
    try:
        import numpy as np
        import pandas as pd

        print("✅ Все необходимые модули загружены успешно!")
        print("🔄 Запуск тор