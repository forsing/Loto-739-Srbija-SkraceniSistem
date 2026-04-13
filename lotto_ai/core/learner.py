"""
Adaptive portfolio learner - v4.0
Learns strategy mix preferences from empirical performance,
not winning-number prediction.
"""
from datetime import datetime
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lotto_ai.core.db import get_session, AdaptiveWeight
from lotto_ai.core.tracker import PredictionTracker
from lotto_ai.config import logger


class AdaptiveLearner:
    def __init__(self):
        self.tracker = PredictionTracker()
        self._initialize_weights()

    def _initialize_weights(self):
        session = get_session()
        try:
            count = session.query(AdaptiveWeight).count()
            if count == 0:
                defaults = [
                    ("coverage_optimized", "coverage_ratio", 1.0, 0.0, 0),
                    ("coverage_optimized", "random_ratio", 0.0, 0.0, 0),
                ]
                for strategy, wtype, value, score, n_obs in defaults:
                    session.add(
                        AdaptiveWeight(
                            updated_at=datetime.now().isoformat(),
                            strategy_name=strategy,
                            weight_type=wtype,
                            weight_value=value,
                            performance_score=score,
                            n_observations=n_obs,
                        )
                    )
                session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Error initializing weights: {e}")
        finally:
            session.close()

    def get_current_weights(self, strategy_name="coverage_optimized"):
        session = get_session()
        try:
            weights = {}
            for weight_type in ["coverage_ratio", "random_ratio"]:
                row = (
                    session.query(AdaptiveWeight)
                    .filter_by(strategy_name=strategy_name, weight_type=weight_type)
                    .order_by(AdaptiveWeight.updated_at.desc())
                    .first()
                )
                if row:
                    weights[weight_type] = {
                        "value": row.weight_value,
                        "performance": row.performance_score,
                        "n_obs": row.n_observations,
                    }
                else:
                    weights[weight_type] = {
                        "value": 1.0 if weight_type == "coverage_ratio" else 0.0,
                        "performance": 0.0,
                        "n_obs": 0,
                    }
            return weights
        finally:
            session.close()

    def update_weights(self, strategy_name="coverage_optimized", window=50):
        perf = self.tracker.get_strategy_performance(strategy_name, window=window)
        if not perf or perf["n_predictions"] < 10:
            logger.info("Not enough data to update adaptive weights reliably")
            return None

        current = self.get_current_weights(strategy_name)
        current_coverage = current["coverage_ratio"]["value"]
        outperform_rate = perf.get("outperform_random_best_rate", 0.5)

        # Conservative adaptation
        if outperform_rate >= 0.60:
            new_coverage = min(1.0, current_coverage + 0.05)
        elif outperform_rate <= 0.40:
            new_coverage = max(0.50, current_coverage - 0.05)
        else:
            new_coverage = current_coverage

        new_random = 1.0 - new_coverage

        session = get_session()
        try:
            for wtype, value in [
                ("coverage_ratio", new_coverage),
                ("random_ratio", new_random),
            ]:
                session.add(
                    AdaptiveWeight(
                        updated_at=datetime.now().isoformat(),
                        strategy_name=strategy_name,
                        weight_type=wtype,
                        weight_value=value,
                        performance_score=outperform_rate,
                        n_observations=perf["n_predictions"],
                    )
                )
            session.commit()

            logger.info(
                f"Adaptive update: coverage={new_coverage:.2f}, "
                f"random={new_random:.2f}, outperform_rate={outperform_rate:.2f}"
            )

            return {
                "coverage_ratio": new_coverage,
                "random_ratio": new_random,
                "performance_score": outperform_rate,
                "n_observations": perf["n_predictions"],
            }
        except Exception as e:
            session.rollback()
            logger.error(f"Error updating weights: {e}")
            return None
        finally:
            session.close()