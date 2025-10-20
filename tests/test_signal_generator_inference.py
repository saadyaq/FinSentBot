import numpy as np
import pytest

from TradingLogic.SignalGenerator.signal_generator import SignalGenerator, TradingSignal
from TradingLogic.SignalGenerator.train import ModelTrainer


@pytest.fixture
def trained_checkpoint(tmp_path, sample_csv):
    trainer = ModelTrainer(model_type="mlp", device="cpu")
    train_loader, val_loader = trainer.prepare_data(
        sample_csv,
        test_size=0.3,
        batch_size=4,
        sequence_length=3,
        split_strategy="random",
    )
    trainer.train(train_loader, val_loader, epochs=2, learning_rate=5e-3)
    ckpt_dir = tmp_path / "signal_model"
    ckpt_path = trainer.save_model(ckpt_dir)
    return ckpt_path


def test_signal_generator_prediction(trained_checkpoint):
    generator = SignalGenerator(str(trained_checkpoint), confidence_threshold=0.0)

    market_data = {
        "sentiment_score": 0.15,
        "price": 120.0,
        "previous_price": 118.0,
        "variation": (120.0 - 118.0) / 118.0,
        "price_future": 121.0,
    }

    signal = generator.generate_signal("AAA", market_data)
    assert isinstance(signal, TradingSignal)
    assert signal.action in {"BUY", "HOLD", "SELL"}
    assert 0.0 <= signal.confidence <= 1.0
    assert signal.price == pytest.approx(120.0)

    # confidence gating should downgrade action when threshold too high
    cautious_generator = SignalGenerator(str(trained_checkpoint), confidence_threshold=1.01)
    hold_signal = cautious_generator.generate_signal("AAA", market_data)
    assert hold_signal.action == "HOLD"
    assert hold_signal.reasoning.startswith("Confiance trop faible")
