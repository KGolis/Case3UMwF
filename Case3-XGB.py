import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from backtesting import Backtest, Strategy

# Pobranie i przygotowanie danych
def prepare_data(ticker):
    df = yf.download(ticker, start="2020-01-01", end="2024-05-07", auto_adjust=True, progress=False)

    # MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns={'Adj Close': 'Close'})

    # Feature Engineering
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()

    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + gain / loss))

    # Zmienna celu: 1 jeśli cena jutro wzrośnie, 0 jeśli spadnie
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    return df.dropna()



data = prepare_data("MSFT")

# Podział na zbiór treningowy i testowy
train = data[data.index < "2024-01-01"].copy()
test = data[data.index >= "2024-01-01"].copy()
features = ['SMA_10', 'SMA_50', 'RSI', 'Open', 'Volume']

# Automatyczna Optymalizacja Modelu XGBoost (GridSearch)
xgb = XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False)

param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2]
}

# Szukamy najlepszych ustawień
grid_search = GridSearchCV(xgb, param_grid, cv=3, n_jobs=-1)
grid_search.fit(train[features], train['Target'])

best_model = grid_search.best_estimator_
print(f"   -> Znaleziono najlepszy model: {grid_search.best_params_}")

# Generujemy prawdopodobieństwa dla zbioru testowego
test['Probability'] = best_model.predict_proba(test[features])[:, 1]


# Definicja Strategii do Optymalizacji
class AutoOptimizedStrategy(Strategy):
    # Wartości startowe (zostaną nadpisane przez optimize)
    threshold = 0.50
    sl_pct = 0.02
    tp_pct = 0.05

    def init(self):
        # Rejestrujemy wskaźnik prawdopodobieństwa
        self.proba = self.I(lambda x: x, self.data.Probability)

    def next(self):
        price = self.data.Close[-1]

        # Jeśli nie mamy pozycji i pewność modelu przekracza próg
        if not self.position and self.proba[-1] > self.threshold:
            # Obliczamy ceny wyjścia dynamicznie
            stop_loss = price * (1 - self.sl_pct)
            take_profit = price * (1 + self.tp_pct)

            # Otwieramy pozycję
            self.buy(sl=stop_loss, tp=take_profit)


# Backtest
#cash 10k , prowizja 0,05%
bt = Backtest(test, AutoOptimizedStrategy, cash=10000, commission=.0005)

# Optymalizacja
stats, heatmap = bt.optimize(
    threshold=[x / 100 for x in range(40, 75, 5)],
    sl_pct=[x / 100 for x in range(1, 10, 1)],
    tp_pct=[x / 100 for x in range(2, 16, 2)],
    maximize='Return [%]',
    constraint=lambda p: p.tp_pct > p.sl_pct,
    return_heatmap=True
)

# Wyniki
print(f"Próg wejścia (Threshold): {stats['_strategy'].threshold:.2f}")
print(f"Stop Loss (%):            {stats['_strategy'].sl_pct * 100:.1f}%")
print(f"Take Profit (%):          {stats['_strategy'].tp_pct * 100:.1f}%")


print("Backtest: ")
print(stats)

# Wizualizacja
bt.plot()