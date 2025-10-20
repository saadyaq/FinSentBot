import numpy as np
import pandas as pd
import pytest
import torch

from TradingLogic.SignalGenerator.models import LSTMSignalGenerator, SimpleMLP
from TradingLogic.SignalGenerator.train import TradingDataset, ModelTrainer


def test_trading_dataset_sequence_shape(sample_dataframe):
    dataset = TradingDataset(sample_dataframe, sequence_length=4, use_sequences=True)
    seq, label = dataset[0]

    assert len(dataset) == len(sample_dataframe)
    assert seq.shape == (4, 4)  # sequence_length x num_features
    assert seq.dtype == torch.float32
    assert isinstance(label.item(), int)

    # feature scaling should produce approximately zero mean
    feature_means = dataset.X.mean(axis=0)
    assert np.all(np.abs(feature_means) < 1e-6)
    assert set(dataset.label_encoder.classes_) == {"BUY", "HOLD", "SELL"}


def test_trading_dataset_flat_mode(sample_dataframe):
    dataset = TradingDataset(sample_dataframe, use_sequences=False)
    features, label = dataset[0]
    assert features.shape == (4,)
    assert features.dtype == torch.float32
    assert isinstance(label.item(), int)


@pytest.mark.parametrize("model_type", ["mlp", "lstm"])
def test_model_trainer_prepare_and_train(tmp_path, sample_csv, model_type):
    trainer = ModelTrainer(model_type=model_type, device="cpu")
    train_loader, val_loader = trainer.prepare_data(
        sample_csv,
        test_size=0.3,
        batch_size=2,
        sequence_length=3,
        split_strategy="random",
    )

    assert len(train_loader) > 0
    assert len(val_loader) > 0
    history = trainer.train(train_loader, val_loader, epochs=1, learning_rate=5e-3)
    assert len(history["train_loss"]) == 1
    assert trainer.model is not None

    # save and ensure checkpoint contains scaler and label encoder
    output_dir = tmp_path / f"checkpoint_{model_type}"
    ckpt_path = trainer.save_model(output_dir)
    saved = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert "scaler" in saved and "label_encoder" in saved


def test_model_trainer_time_split(sample_csv):
    trainer = ModelTrainer(model_type="mlp", device="cpu")
    train_loader, val_loader = trainer.prepare_data(
        sample_csv,
        test_size=0.4,
        batch_size=4,
        sequence_length=2,
        split_strategy="time",
    )
    assert len(train_loader.dataset) + len(val_loader.dataset) == len(trainer.sample_info)
    assert len(val_loader.dataset) > 0


def test_model_trainer_symbol_split(sample_csv):
    trainer = ModelTrainer(model_type="mlp", device="cpu")
    train_loader, val_loader = trainer.prepare_data(
        sample_csv,
        test_size=0.3,
        batch_size=4,
        sequence_length=2,
        split_strategy="symbol",
        val_symbols=["EEE", "FFF"],
    )
    val_symbols = {info[0] for idx, info in enumerate(trainer.sample_info)
                   if idx in val_loader.dataset.indices}
    assert val_symbols <= {"EEE", "FFF"}


def test_model_forward_shapes():
    mlp = SimpleMLP(input_dim=4)
    logits, conf = mlp(torch.zeros(3, 4))
    assert logits.shape == (3, 3)
    assert conf.shape == (3, 1)
    assert torch.all((conf >= 0) & (conf <= 1))

    lstm = LSTMSignalGenerator(input_dim=4, hidden_dim=16, num_layers=1)
    logits, conf = lstm(torch.zeros(3, 5, 4))
    assert logits.shape == (3, 3)
    assert conf.shape == (3, 1)
    assert torch.all((conf >= 0) & (conf <= 1))
