
from .signal_generator import SignalGenerator, TradingSignal

ModelTrainer = None
train_model_cli = None
LSTMSignalGenerator = None
SimpleMLP = None

try:
    from .train import ModelTrainer, train_model_cli  # type: ignore
    from .models import LSTMSignalGenerator, SimpleMLP  # type: ignore
except ImportError:
    # Ces composants ne sont nécessaires que pour l'entraînement
    pass

__all__ = [
    "SignalGenerator",
    "TradingSignal",
]

if ModelTrainer is not None:
    __all__.extend(
        [
            "ModelTrainer",
            "train_model_cli",
            "LSTMSignalGenerator",
            "SimpleMLP",
        ]
    )
