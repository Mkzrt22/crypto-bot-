"""
LSTM + Transformer model for crypto price prediction.
Understands temporal patterns that tree models miss.

Requires: torch (already installed)
"""
import logging
import numpy as np
import os
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch not available")


class LSTMTransformer(nn.Module):
    """Hybrid LSTM + Transformer for time series prediction."""

    def __init__(self, input_size: int, hidden_size: int = 128,
                 num_layers: int = 2, nhead: int = 4, dropout: float = 0.2):
        super().__init__()
        self.hidden_size = hidden_size

        # LSTM for local patterns
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )

        # Transformer for global attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size * 2,  # bidirectional
            nhead=nhead,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Output head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        attn_out = self.transformer(lstm_out)
        # Use last timestep
        out = attn_out[:, -1, :]
        return self.classifier(out).squeeze(-1)


class DeepLearningAgent:
    """LSTM+Transformer agent that complements tree models."""

    def __init__(self, config):
        self.config = config
        self.model = None
        self.is_trained = False
        self.last_trained = None
        self.last_accuracy = 0.0
        self.seq_len = 48  # Look back 48 candles (2 days of 1h) — captures longer patterns
        self.device = "cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu"
        self._input_size = None
        self._feature_mean = None
        self._feature_std = None
        self._retrain_hours = 4
        self.model_path = "models/lstm_transformer.pt"
        self.stats_path = "models/lstm_stats.npz"   # NEW: normalisation stats file
        os.makedirs("models", exist_ok=True)
        logger.info(f"DeepLearningAgent device: {self.device}")
        # Try to restore normalisation stats from disk (model itself is loaded in predict-time)
        self._try_load_stats()

    def _try_load_stats(self):
        """Restore feature mean/std (and input_size) from disk if available."""
        if not HAS_TORCH:
            return
        try:
            if os.path.exists(self.stats_path):
                data = np.load(self.stats_path, allow_pickle=False)
                self._feature_mean = data.get("mean")
                self._feature_std = data.get("std")
                self._input_size = int(data.get("input_size", 0)) or None
                logger.info("LSTM: restored feature normalisation stats from disk")
        except Exception as e:
            logger.debug(f"LSTM stats load failed: {e}")

    def _save_stats(self):
        """Persist normalisation stats alongside the model."""
        try:
            if self._feature_mean is not None and self._feature_std is not None:
                np.savez(
                    self.stats_path,
                    mean=self._feature_mean,
                    std=self._feature_std,
                    input_size=np.array(self._input_size or 0),
                )
        except Exception as e:
            logger.debug(f"LSTM stats save failed: {e}")

    def needs_retraining(self) -> bool:
        if not self.is_trained or self.last_trained is None:
            return True
        elapsed = (datetime.now() - self.last_trained).total_seconds() / 3600
        return elapsed >= self._retrain_hours

    def _prepare_sequences(self, X: np.ndarray, y: np.ndarray):
        """Convert flat features to sequences."""
        sequences = []
        labels = []
        for i in range(self.seq_len, len(X)):
            sequences.append(X[i - self.seq_len:i])
            labels.append(y[i])
        return np.array(sequences, dtype=np.float32), np.array(labels, dtype=np.float32)

    def train(self, df, feature_cols: list) -> float:
        if not HAS_TORCH:
            return 0.0

        try:
            clean = df[feature_cols + ["label"]].dropna()
            if len(clean) < self.seq_len + 100:
                logger.warning(f"Not enough data for LSTM: {len(clean)}")
                return 0.0

            clean["label"] = clean["label"].astype(int)
            X = clean[feature_cols].values.astype(np.float32)
            y = clean["label"].values.astype(np.float32)

            # Normalize
            self._feature_mean = X.mean(axis=0)
            self._feature_std = X.std(axis=0) + 1e-8
            X_norm = (X - self._feature_mean) / self._feature_std
            X_norm = np.nan_to_num(X_norm, nan=0.0, posinf=1.0, neginf=-1.0)

            # Create sequences
            X_seq, y_seq = self._prepare_sequences(X_norm, y)

            # Train/val split (temporal)
            split = int(len(X_seq) * 0.8)
            X_tr, X_val = X_seq[:split], X_seq[split:]
            y_tr, y_val = y_seq[:split], y_seq[split:]

            # Build model
            self._input_size = len(feature_cols)
            self.model = LSTMTransformer(
                input_size=self._input_size,
                hidden_size=128,
                num_layers=2,
                nhead=4,
                dropout=0.2,
            ).to(self.device)

            # Training
            optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
            criterion = nn.BCELoss()

            X_tr_t = torch.FloatTensor(X_tr).to(self.device)
            y_tr_t = torch.FloatTensor(y_tr).to(self.device)
            X_val_t = torch.FloatTensor(X_val).to(self.device)
            y_val_t = torch.FloatTensor(y_val).to(self.device)

            dataset = TensorDataset(X_tr_t, y_tr_t)
            loader = DataLoader(dataset, batch_size=64, shuffle=False)

            best_val_acc = 0
            patience = 10
            patience_counter = 0

            for epoch in range(100):  # Max 100 epochs
                self.model.train()
                total_loss = 0

                for X_batch, y_batch in loader:
                    optimizer.zero_grad()
                    pred = self.model(X_batch)
                    loss = criterion(pred, y_batch)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    total_loss += loss.item()

                # Validation
                self.model.eval()
                with torch.no_grad():
                    val_pred = self.model(X_val_t)
                    val_pred_class = (val_pred > 0.5).float()
                    val_acc = (val_pred_class == y_val_t).float().mean().item()
                    val_loss = criterion(val_pred, y_val_t)

                scheduler.step(val_loss)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(self.model.state_dict(), self.model_path)
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    logger.info(f"LSTM early stop at epoch {epoch}")
                    break

                if epoch % 10 == 0:
                    logger.info(f"LSTM epoch {epoch}: loss={total_loss/len(loader):.4f} val_acc={val_acc:.3f}")

            # Load best model
            if os.path.exists(self.model_path):
                # weights_only=True for security (required on PyTorch >= 2.6)
                try:
                    state = torch.load(self.model_path, map_location=self.device, weights_only=True)
                except TypeError:
                    # Older PyTorch without weights_only arg
                    state = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(state)

            self.last_accuracy = best_val_acc
            self.is_trained = True
            self.last_trained = datetime.now()
            # Persist normalisation stats alongside the model
            self._save_stats()
            logger.info(f"LSTM+Transformer trained: best_val_acc={best_val_acc:.3f}")
            return best_val_acc

        except Exception as e:
            logger.error(f"LSTM training failed: {e}")
            return 0.0

    def predict(self, recent_features: np.ndarray) -> dict:
        """Predict from last seq_len feature vectors."""
        if not HAS_TORCH or not self.is_trained or self.model is None:
            return {"prob_up": 0.5, "signal": "HOLD", "confidence": 0.0}

        try:
            if len(recent_features) < self.seq_len:
                return {"prob_up": 0.5, "signal": "HOLD", "confidence": 0.0}

            # Take last seq_len rows
            X = recent_features[-self.seq_len:].astype(np.float32)

            # Normalize
            if self._feature_mean is not None:
                X = (X - self._feature_mean) / self._feature_std
                X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)

            X_t = torch.FloatTensor(X).unsqueeze(0).to(self.device)

            self.model.eval()
            with torch.no_grad():
                prob = self.model(X_t).item()

            signal = "BUY" if prob > 0.55 else "SELL" if prob < 0.45 else "HOLD"

            return {
                "prob_up": prob,
                "signal": signal,
                "confidence": abs(prob - 0.5) * 2,  # 0-1 scale
                "accuracy": self.last_accuracy,
            }

        except Exception as e:
            logger.debug(f"LSTM predict error: {e}")
            return {"prob_up": 0.5, "signal": "HOLD", "confidence": 0.0}
