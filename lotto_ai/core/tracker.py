"""
Prediction / portfolio tracking - v4.0
Empirical baseline comparison, not theoretical-only.
"""
from datetime import datetime
import json
import numpy as np
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lotto_ai.core.db import get_session, Prediction, PredictionResult, PlayedTicket, Draw
from lotto_ai.config import (
    logger, PRIZE_TABLE, TICKET_COST,
    RANDOM_BASELINE_PORTFOLIOS, RNG_SEED
)
from lotto_ai.core.coverage_optimizer import generate_random_portfolio
from lotto_ai.core.math_engine import evaluate_portfolio_once


class PredictionTracker:
    def save_prediction(self, target_draw_date, strategy_name, tickets,
                        model_version="4.0", metadata=None):
        session = get_session()
        try:
            meta_str = json.dumps(metadata or {}, default=str)

            prediction = Prediction(
                created_at=datetime.now().isoformat(),
                target_draw_date=target_draw_date,
                strategy_name=strategy_name,
                model_version=model_version,
                portfolio_size=len(tickets),
                tickets=json.dumps(tickets),
                model_metadata=meta_str,
                evaluated=False
            )
            session.add(prediction)
            session.commit()
            pred_id = prediction.prediction_id
            logger.info(f"Saved portfolio {pred_id} for {target_draw_date}")
            return pred_id
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving prediction: {e}")
            return None
        finally:
            session.close()

    def evaluate_prediction(self, prediction_id, actual_numbers, n_random_baselines=200):
        session = get_session()
        try:
            prediction = session.query(Prediction).filter_by(
                prediction_id=prediction_id
            ).first()
            if not prediction:
                logger.error(f"Prediction {prediction_id} not found")
                return None

            tickets = json.loads(prediction.tickets)
            actual_set = set(actual_numbers)

            outcome = evaluate_portfolio_once(tickets, actual_set, prize_table=PRIZE_TABLE)
            empirical = self._empirical_random_baseline(
                portfolio_size=len(tickets),
                actual_numbers=actual_numbers,
                n_random_baselines=n_random_baselines,
            )

            result_payload = {
                "prediction_id": prediction_id,
                "best_match": outcome["best_match"],
                "total_matches": int(sum(outcome["ticket_matches"])),
                "prize_value": float(outcome["total_prize"]),
                "ticket_matches": outcome["ticket_matches"],
                "empirical_random_baseline": empirical,
            }

            result = PredictionResult(
                prediction_id=prediction_id,
                actual_numbers=json.dumps(actual_numbers),
                evaluated_at=datetime.now().isoformat(),
                best_match=outcome["best_match"],
                total_matches=int(sum(outcome["ticket_matches"])),
                prize_value=float(outcome["total_prize"]),
                ticket_matches=json.dumps(result_payload),
            )
            session.add(result)

            prediction.evaluated = True
            session.commit()

            logger.info(
                f"Evaluated portfolio {prediction_id}: "
                f"best={outcome['best_match']}/7, "
                f"percentile_vs_random={empirical['best_match_percentile']:.1f}"
            )
            return result_payload
        except Exception as e:
            session.rollback()
            logger.error(f"Error evaluating prediction: {e}")
            return None
        finally:
            session.close()

    def auto_evaluate_pending(self):
        session = get_session()
        try:
            pending = session.query(Prediction).filter_by(evaluated=False).all()
            if not pending:
                return 0

            evaluated_count = 0
            for pred in pending:
                draw = session.query(Draw).filter_by(draw_date=pred.target_draw_date).first()
                if draw:
                    self.evaluate_prediction(pred.prediction_id, draw.get_numbers())
                    evaluated_count += 1
            return evaluated_count
        finally:
            session.close()

    def _empirical_random_baseline(self, portfolio_size, actual_numbers, n_random_baselines=200):
        actual_set = set(actual_numbers)
        best_matches = []
        prize_values = []

        for i in range(n_random_baselines):
            random_portfolio, _ = generate_random_portfolio(
                n_tickets=portfolio_size,
                rng_seed=RNG_SEED + i
            )
            outcome = evaluate_portfolio_once(random_portfolio, actual_set, prize_table=PRIZE_TABLE)
            best_matches.append(outcome["best_match"])
            prize_values.append(outcome["total_prize"])

        return {
            "n_random_baselines": n_random_baselines,
            "random_best_match_mean": float(np.mean(best_matches)),
            "random_best_match_std": float(np.std(best_matches)),
            "random_prize_mean": float(np.mean(prize_values)),
            "random_prize_std": float(np.std(prize_values)),
        }

    def get_strategy_performance(self, strategy_name, window=50):
        session = get_session()
        try:
            rows = (
                session.query(PredictionResult, Prediction)
                .join(Prediction, Prediction.prediction_id == PredictionResult.prediction_id)
                .filter(Prediction.strategy_name == strategy_name, Prediction.evaluated == True)
                .order_by(PredictionResult.evaluated_at.desc())
                .limit(window)
                .all()
            )

            if not rows:
                return None

            best_matches = []
            prize_values = []
            portfolio_sizes = []
            random_best_means = []
            random_prize_means = []
            outperform_best = 0
            outperform_prize = 0

            for result, pred in rows:
                best_matches.append(result.best_match)
                prize_values.append(result.prize_value or 0.0)
                portfolio_sizes.append(pred.portfolio_size or 0)

                payload = json.loads(result.ticket_matches)
                empirical = payload.get("empirical_random_baseline", {})
                rbm = empirical.get("random_best_match_mean")
                rpm = empirical.get("random_prize_mean")

                if rbm is not None:
                    random_best_means.append(rbm)
                    if result.best_match > rbm:
                        outperform_best += 1

                if rpm is not None:
                    random_prize_means.append(rpm)
                    if (result.prize_value or 0.0) > rpm:
                        outperform_prize += 1

            n = len(best_matches)

            return {
                "n_predictions": n,
                "avg_best_match": float(np.mean(best_matches)),
                "best_ever": int(max(best_matches)),
                "hit_rate_3plus": float(sum(1 for x in best_matches if x >= 3) / n),
                "avg_prize_value": float(np.mean(prize_values)),
                "total_prize_won": float(np.sum(prize_values)),
                "avg_tickets_per_prediction": float(np.mean(portfolio_sizes)) if portfolio_sizes else 0.0,
                "random_best_match_mean": float(np.mean(random_best_means)) if random_best_means else None,
                "random_prize_mean": float(np.mean(random_prize_means)) if random_prize_means else None,
                "outperform_random_best_rate": float(outperform_best / n) if n > 0 else 0.0,
                "outperform_random_prize_rate": float(outperform_prize / n) if n > 0 else 0.0,
            }
        except Exception as e:
            logger.error(f"Error getting performance: {e}")
            return None
        finally:
            session.close()


class PlayedTicketsTracker:
    def save_played_tickets(self, prediction_id, tickets, draw_date):
        session = get_session()
        try:
            for ticket in tickets:
                played = PlayedTicket(
                    prediction_id=prediction_id,
                    ticket_numbers=json.dumps(ticket),
                    played_at=datetime.now().isoformat(),
                    draw_date=draw_date,
                )
                session.add(played)
            session.commit()
            logger.info(f"Saved {len(tickets)} played tickets for {draw_date}")
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving played tickets: {e}")
        finally:
            session.close()