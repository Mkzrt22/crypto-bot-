import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
import joblib
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

if not HAS_LGBM and not HAS_XGB:
    from sklearn.ensemble import GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression

TOP_N_FEATURES = 40
PURGE_GAP = 5  # Gap between train and test to avoid lookahead leakage


class TradingAgent:
    def __init__(self, config):
        self.config = config
        self.models = []
        self.stacker = None           # Meta-learner for stacking
        self.scaler = None
        self.feature_cols = []
        self.selected_features = []
        self.feature_importances = {}
        self._feature_importance_history = {}  # name -> liste des scores sur derniers retrains
        self._excluded_features = set()  # Features exclues car toujours faibles
        self.is_trained = False
        self.last_trained = None
        self.last_accuracy = 0.0
        self.best_params = {}
        self.model_dir = "models"
        self.model_weights = {}  # name → weight (based on recent performance)
        os.makedirs(self.model_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, df, feature_cols):
        self.feature_cols = feature_cols
        logger.info(f"Training: {len(feature_cols)} features, stacking + calibration")
        acc = self._walk_forward_train(df, feature_cols)
        self.last_accuracy = acc
        self.is_trained = True
        self.last_trained = datetime.now()
        logger.info(f"Training complete — accuracy: {acc:.3f} ({len(self.selected_features)} features)")
        return acc

    # ------------------------------------------------------------------
    # Temporal weighting
    # ------------------------------------------------------------------

    def _get_sample_weights(self, n_samples):
        """Recent samples get more weight. Exponential decay."""
        decay = 0.995  # each older sample gets 0.5% less weight
        weights = np.array([decay ** (n_samples - i - 1) for i in range(n_samples)])
        # Normalize so mean weight = 1
        weights = weights / weights.mean()
        return weights

    # ------------------------------------------------------------------
    # Feature selection
    # ------------------------------------------------------------------

    def _select_features(self, X, y, feature_cols, sample_weight=None):
        try:
            if HAS_LGBM:
                clf = LGBMClassifier(n_estimators=100, max_depth=4, verbose=-1, random_state=42)
                clf.fit(X, y, sample_weight=sample_weight)
            else:
                clf = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
                clf.fit(X, y, sample_weight=sample_weight)

            importances = clf.feature_importances_
            indices = np.argsort(importances)[::-1][:TOP_N_FEATURES]
            selected = [feature_cols[i] for i in indices]

            self.feature_importances = {
                feature_cols[i]: float(importances[i])
                for i in indices
            }
            top5 = [(feature_cols[i], round(float(importances[i]), 1)) for i in indices[:5]]

            # Tracker l historique des importances sur les derniers retrains
            for feat, score in self.feature_importances.items():
                if feat not in self._feature_importance_history:
                    self._feature_importance_history[feat] = []
                self._feature_importance_history[feat].append(score)
                self._feature_importance_history[feat] = self._feature_importance_history[feat][-5:]

            # Exclure features toujours en bas apres 5 retrains
            newly_excluded = []
            for feat, history in self._feature_importance_history.items():
                if len(history) >= 5:
                    avg_score = sum(history) / len(history)
                    global_avg = sum(self.feature_importances.values()) / max(len(self.feature_importances), 1)
                    if avg_score < global_avg * 0.1 and feat not in self._excluded_features:
                        self._excluded_features.add(feat)
                        newly_excluded.append(feat)

            if newly_excluded:
                logger.info(f"Features exclues (toujours faibles): {newly_excluded}")

            # Filtrer les features exclues de la selection
            selected = [f for f in selected if f not in self._excluded_features]
            logger.info(f"Feature selection: {len(selected)}/{len(feature_cols)} — top 5: {top5}")
            if self._excluded_features:
                logger.info(f"Features exclues actives: {len(self._excluded_features)}")
            return selected
        except Exception as e:
            logger.warning(f"Feature selection failed: {e}")
            return feature_cols

    # ------------------------------------------------------------------
    # Walk-forward with stacking
    # ------------------------------------------------------------------

    def _walk_forward_train(self, df, feature_cols):
        df_clean = df[feature_cols + ["label"]].dropna()
        df_clean["label"] = df_clean["label"].astype(int)
        n = len(df_clean)

        if n < self.config.MIN_TRAIN_SAMPLES + 50:
            return self._train_single(df_clean, feature_cols)

        # Feature selection with temporal weighting
        sel_end = int(n * 0.7)
        X_sel = df_clean[feature_cols].iloc[:sel_end].values  # numpy ok pour selection
        y_sel = df_clean["label"].iloc[:sel_end].values
        w_sel = self._get_sample_weights(len(y_sel))

        scaler_sel = StandardScaler()
        X_sel_s = scaler_sel.fit_transform(X_sel)
        self.selected_features = self._select_features(X_sel_s, y_sel, feature_cols, w_sel)

        sel_cols = [c for c in self.selected_features if c in df_clean.columns]
        df_sel = df_clean[sel_cols + ["label"]]

        train_size = int(n * 0.6)
        test_size = int(n * 0.1)
        step = test_size

        accuracies = []
        last_models = None
        last_scaler = None
        last_stacker = None
        fold = 0
        start = 0

        # Collect stacking meta-features
        meta_X_all = []
        meta_y_all = []

        while start + train_size + test_size <= n:
            fold += 1
            train_end = start + train_size
            test_end = min(train_end + test_size, n)

            X_tr = df_sel[sel_cols].iloc[start:train_end]  # DataFrame pour feature names
            y_tr = df_sel["label"].iloc[start:train_end].values
            # Purge gap: skip PURGE_GAP rows between train and test
            gap_end = min(train_end + PURGE_GAP, n)
            X_te = df_sel[sel_cols].iloc[gap_end:test_end]  # DataFrame pour feature names
            y_te = df_sel["label"].iloc[gap_end:test_end].values
            if len(X_te) < 10:
                start += step
                continue

            # Temporal weighting
            w_tr = self._get_sample_weights(len(y_tr))

            scaler = StandardScaler()
            import pandas as _pd
            X_tr_s = _pd.DataFrame(scaler.fit_transform(X_tr), columns=sel_cols)
            X_te_s = _pd.DataFrame(scaler.transform(X_te), columns=sel_cols)

            models = self._build_models(X_tr_s, y_tr, w_tr, fold == 1)
            if not models:
                start += step
                continue

            # Collect meta-features for stacking
            meta_features = self._get_meta_features(models, X_te_s)
            meta_X_all.append(meta_features)
            meta_y_all.append(y_te)

            # Evaluate per-model accuracy for adaptive weights
            for name, model in models:
                y_m = model.predict(X_te_s)
                m_acc = accuracy_score(y_te, y_m)
                if name not in self.model_weights:
                    self.model_weights[name] = []
                self.model_weights[name].append(m_acc)

            # Evaluate base ensemble
            y_pred = self._ensemble_predict_class(models, X_te_s)
            acc = accuracy_score(y_te, y_pred)
            accuracies.append(acc)

            last_models = models
            last_scaler = scaler

            logger.info(f"Walk-forward fold {fold}: acc={acc:.3f}")
            start += step

        # Train stacker on all meta-features
        if meta_X_all and last_models:
            meta_X = np.vstack(meta_X_all)
            meta_y = np.concatenate(meta_y_all)
            last_stacker = self._train_stacker(meta_X, meta_y)

        if last_models:
            self.models = last_models
            self.scaler = last_scaler
            self.stacker = last_stacker

            # Calibration enabled — improves probability reliability for Kelly sizing
            try:
                self._calibrate_models(df_sel, sel_cols)
            except Exception as _e:
                logger.debug(f"Calibration skipped: {_e}")

            self._save_model(sel_cols)

        avg_acc = np.mean(accuracies) if accuracies else 0.0
        labels = df_clean["label"].values
        logger.info(f"Walk-forward: {len(accuracies)} folds, avg accuracy: {avg_acc:.3f}")
        # Log per-model weights
        for name, accs in self.model_weights.items():
            logger.info(f"  {name}: avg_acc={np.mean(accs):.3f} (weight={np.mean(accs[-3:]):.3f})")
        logger.info(f"Labels: UP={(labels==1).mean():.1%} DOWN={(labels==0).mean():.1%}")
        return avg_acc

    def _train_single(self, df_clean, feature_cols):
        X = df_clean[feature_cols].values
        y = df_clean["label"].values
        split = int(len(X) * 0.8)
        w = self._get_sample_weights(split)

        self.scaler = StandardScaler()
        X_tr = self.scaler.fit_transform(X[:split])
        X_te = self.scaler.transform(X[split:])

        self.selected_features = self._select_features(X_tr, y[:split], feature_cols, w)
        sel_idx = [feature_cols.index(f) for f in self.selected_features if f in feature_cols]
        X_tr = X_tr[:, sel_idx]
        X_te = X_te[:, sel_idx]

        self.models = self._build_models(X_tr, y[:split], w, use_optuna=True)
        if not self.models:
            return 0.0

        y_pred = self._ensemble_predict_class(self.models, X_te)
        acc = accuracy_score(y[split:], y_pred)
        self._save_model(self.selected_features)
        return acc

    # ------------------------------------------------------------------
    # Model building with temporal weights
    # ------------------------------------------------------------------

    def _build_models(self, X_tr, y_tr, sample_weight=None, use_optuna=False):
        models = []

        if HAS_OPTUNA and use_optuna and len(X_tr) > 200:
            params = self._optuna_tune(X_tr, y_tr, sample_weight)
        else:
            params = self.best_params or {}

        if HAS_LGBM:
            lgbm = LGBMClassifier(
                n_estimators=params.get("n_estimators", 200),
                max_depth=params.get("max_depth", 5),
                learning_rate=params.get("lr", 0.05),
                subsample=params.get("subsample", 0.8),
                colsample_bytree=params.get("colsample", 0.8),
                min_child_samples=params.get("min_child", 20),
                reg_alpha=params.get("alpha", 0.1),
                reg_lambda=params.get("lambda", 0.1),
                random_state=42, verbose=-1, objective="binary",
            )
            lgbm.fit(X_tr, y_tr, sample_weight=sample_weight)
            models.append(("lgbm", lgbm))

        if HAS_XGB and self.config.ENSEMBLE_ENABLED:
            xgb = XGBClassifier(
                n_estimators=params.get("n_estimators", 200),
                max_depth=params.get("max_depth", 5),
                learning_rate=params.get("lr", 0.05),
                subsample=params.get("subsample", 0.8),
                colsample_bytree=params.get("colsample", 0.8),
                reg_alpha=params.get("alpha", 0.1),
                reg_lambda=params.get("lambda", 0.1),
                random_state=43, use_label_encoder=False,
                eval_metric="logloss", verbosity=0,
                objective="binary:logistic",
            )
            xgb.fit(X_tr, y_tr, sample_weight=sample_weight)
            models.append(("xgb", xgb))

        # CatBoost
        if HAS_CATBOOST and self.config.ENSEMBLE_ENABLED:
            cb = CatBoostClassifier(
                iterations=params.get("n_estimators", 200),
                depth=min(params.get("max_depth", 5), 8),
                learning_rate=params.get("lr", 0.05),
                l2_leaf_reg=params.get("lambda", 0.1),
                random_seed=44, verbose=0,
                loss_function="Logloss",
            )
            cb.fit(X_tr, y_tr, sample_weight=sample_weight)
            models.append(("catboost", cb))

        if not models:
            gb = GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42)
            gb.fit(X_tr, y_tr, sample_weight=sample_weight)
            models.append(("gbm", gb))

        return models

    # ------------------------------------------------------------------
    # Stacking meta-learner
    # ------------------------------------------------------------------

    def _get_meta_features(self, models, X):
        """Get probability predictions from each base model as meta-features."""
        meta = []
        for name, model in models:
            proba = model.predict_proba(X)
            meta.append(proba[:, 1])  # prob of UP class
        return np.column_stack(meta)

    def _train_stacker(self, meta_X, meta_y):
        """Train logistic regression on meta-features (stacking)."""
        try:
            stacker = LogisticRegression(random_state=42, max_iter=1000)
            stacker.fit(meta_X, meta_y)
            acc = accuracy_score(meta_y, stacker.predict(meta_X))
            logger.info(f"Stacker trained: meta-accuracy={acc:.3f}")
            return stacker
        except Exception as e:
            logger.warning(f"Stacker training failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def _calibrate_models(self, df_sel, sel_cols):
        """Calibrate model probabilities using isotonic regression."""
        try:
            n = len(df_sel)
            cal_start = int(n * 0.7)
            X_cal = df_sel[sel_cols].iloc[cal_start:].values
            y_cal = df_sel["label"].iloc[cal_start:].values

            if len(X_cal) < 50:
                return

            X_cal_s = self.scaler.transform(X_cal)

            calibrated_models = []
            for name, model in self.models:
                try:
                    cal = CalibratedClassifierCV(model, cv="prefit", method="isotonic")
                    cal.fit(X_cal_s, y_cal)
                    calibrated_models.append((f"{name}_cal", cal))
                    logger.info(f"Calibrated {name}")
                except Exception as e:
                    logger.debug(f"Calibration failed for {name}: {e}")
                    calibrated_models.append((name, model))

            self.models = calibrated_models

        except Exception as e:
            logger.warning(f"Calibration failed: {e}")

    # ------------------------------------------------------------------
    # Optuna
    # ------------------------------------------------------------------

    def _optuna_tune(self, X, y, sample_weight=None, n_trials=150):
        logger.info(f"Optuna tuning ({n_trials} trials)...")
        split = int(len(X) * 0.75)
        X_tr, X_val = X[:split], X[split:]
        y_tr, y_val = y[:split].astype(int), y[split:].astype(int)
        w_tr = sample_weight[:split] if sample_weight is not None else None

        def objective(trial):
            p = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 5),
                "learning_rate": trial.suggest_float("lr", 0.01, 0.15, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample", 0.6, 1.0),
                "min_child_samples": trial.suggest_int("min_child", 10, 50),
                "reg_alpha": trial.suggest_float("alpha", 1e-2, 2.0, log=True),
                "reg_lambda": trial.suggest_float("lambda", 1e-3, 1.0, log=True),
            }
            if HAS_LGBM:
                clf = LGBMClassifier(**p, random_state=42, verbose=-1, objective="binary")
                clf.fit(X_tr, y_tr, sample_weight=w_tr)
            else:
                clf = GradientBoostingClassifier(
                    n_estimators=p["n_estimators"], max_depth=p["max_depth"],
                    learning_rate=p["learning_rate"], subsample=p["subsample"],
                    random_state=42)
                clf.fit(X_tr, y_tr, sample_weight=w_tr)
            return accuracy_score(y_val, clf.predict(X_val))

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False, n_jobs=6)
        self.best_params = study.best_params
        logger.info(f"Optuna best: acc={study.best_value:.3f}")
        return self.best_params

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _ensemble_predict_class(self, models, X):
        """Weighted ensemble prediction based on recent model accuracy."""
        all_proba = []
        weights = []
        for name, model in models:
            all_proba.append(model.predict_proba(X)[:, 1])
            # Utiliser les 3 derniers scores de precision comme poids
            base_name = name.replace("_cal", "")
            recent_accs = self.model_weights.get(base_name, [0.5])
            weight = float(np.mean(recent_accs[-3:]))
            weights.append(max(weight, 0.1))  # Poids min 0.1

        if not all_proba:
            return np.array([0])

        # Moyenne ponderee
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normaliser
        avg = np.average(all_proba, axis=0, weights=weights)
        return (avg > 0.5).astype(int)

    def predict(self, features):
        if not self.is_trained or not self.models:
            return {"action": 0, "confidence": 0.0, "signal": "HOLD", "source": "none"}

        import pandas as _pd
        X = features.reshape(1, -1)

        # Select features
        sel_names = []
        if self.selected_features and self.feature_cols:
            sel_idx = [self.feature_cols.index(f) for f in self.selected_features if f in self.feature_cols]
            sel_names = [self.feature_cols[i] for i in sel_idx]
            if sel_idx:
                X = X[:, sel_idx]

        if self.scaler:
            X = self.scaler.transform(X)

        # Wrap in DataFrame to avoid feature name warnings
        if sel_names:
            X = _pd.DataFrame(X, columns=sel_names)

        try:
            # Use stacker if available
            if self.stacker:
                meta = self._get_meta_features(self.models, X)
                prob_up = float(self.stacker.predict_proba(meta)[0][1])
                prob_down = 1.0 - prob_up
                source = "stacked"
            else:
                # Weighted average based on recent accuracy
                all_proba = []
                weights = []
                for name, model in self.models:
                    all_proba.append(model.predict_proba(X)[0])
                    # Get average accuracy for this model
                    base_name = name.replace("_cal", "")
                    hist = self.model_weights.get(base_name, [0.5])
                    w = np.mean(hist[-3:])  # Last 3 folds
                    weights.append(w)
                # Normalize weights
                w_sum = sum(weights) or 1
                weights = [w/w_sum for w in weights]
                avg_proba = np.average(all_proba, axis=0, weights=weights)
                logger.debug(f"Ensemble weights: {dict(zip([n for n,_ in self.models], [f'{w:.2f}' for w in weights]))}"   )
                prob_down = float(avg_proba[0])
                prob_up = float(avg_proba[1])
                source = "+".join(n for n, _ in self.models)

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {"action": 0, "confidence": 0.0, "signal": "HOLD", "source": "error"}

        min_conf = self.config.MIN_CONFIDENCE

        if prob_up > min_conf and prob_up > prob_down:
            signal, action, confidence = "BUY", 2, prob_up
        elif prob_down > min_conf and prob_down > prob_up:
            signal, action, confidence = "SELL", 5, prob_down
        else:
            signal, action, confidence = "HOLD", 0, max(prob_up, prob_down)

        return {
            "action": action, "confidence": confidence, "signal": signal,
            "source": source, "model_accuracy": self.last_accuracy,
            "probabilities": {"UP": prob_up, "DOWN": prob_down},
            "calibrated": any("cal" in n for n, _ in self.models),
        }

    def get_feature_importances(self):
        return self.feature_importances

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_model(self, feature_cols):
        joblib.dump({
            "models": self.models,
            "stacker": self.stacker,
            "scaler": self.scaler,
            "feature_cols": self.feature_cols,
            "selected_features": self.selected_features,
            "feature_importances": self.feature_importances,
            "best_params": self.best_params,
        }, f"{self.model_dir}/ensemble.pkl")
        logger.info(f"Model saved — {len(self.models)} models + stacker, {len(self.selected_features)} features")

    def needs_retraining(self):
        if not self.is_trained or self.last_trained is None:
            return True
        elapsed_h = (datetime.now() - self.last_trained).total_seconds() / 3600
        return elapsed_h >= self.config.RETRAIN_EVERY_HOURS

    def load_saved_models(self):
        ens_path = f"{self.model_dir}/ensemble.pkl"
        if os.path.exists(ens_path):
            try:
                data = joblib.load(ens_path)
                self.models = data["models"]
                self.stacker = data.get("stacker")
                self.scaler = data["scaler"]
                self.feature_cols = data.get("feature_cols", [])
                self.selected_features = data.get("selected_features", [])
                self.feature_importances = data.get("feature_importances", {})
                self.best_params = data.get("best_params", {})
                self.is_trained = True
                logger.info(f"Loaded: {[n for n,_ in self.models]}, stacker={'yes' if self.stacker else 'no'}")
                return True
            except Exception as e:
                logger.warning(f"Failed to load: {e}")

        sup_path = f"{self.model_dir}/supervised.pkl"
        if os.path.exists(sup_path):
            try:
                pipeline = joblib.load(sup_path)
                self.scaler = pipeline.named_steps.get("scaler")
                clf = pipeline.named_steps.get("clf")
                if clf:
                    self.models = [("legacy", clf)]
                    self.is_trained = True
            except Exception:
                pass
