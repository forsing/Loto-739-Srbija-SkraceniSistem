"""
Database layer for Loto Serbia - Enhanced with coverage tracking
"""
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lotto_ai.config import DB_PATH, CSV_DRAWS_PATH, NUMBERS_PER_DRAW, logger

Base = declarative_base()


class Draw(Base):
    __tablename__ = 'draws'

    draw_date = Column(String, primary_key=True)
    round_number = Column(Integer, nullable=True)
    n1 = Column(Integer)
    n2 = Column(Integer)
    n3 = Column(Integer)
    n4 = Column(Integer)
    n5 = Column(Integer)
    n6 = Column(Integer)
    n7 = Column(Integer)

    def get_numbers(self):
        return [self.n1, self.n2, self.n3, self.n4, self.n5, self.n6, self.n7]


class Prediction(Base):
    __tablename__ = 'predictions'

    prediction_id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(String, nullable=False)
    target_draw_date = Column(String, nullable=False)
    strategy_name = Column(String, nullable=False)
    model_version = Column(String)
    portfolio_size = Column(Integer)
    tickets = Column(Text, nullable=False)
    model_metadata = Column(Text)
    evaluated = Column(Boolean, default=False)

    results = relationship("PredictionResult", back_populates="prediction")


class PredictionResult(Base):
    __tablename__ = 'prediction_results'

    result_id = Column(Integer, primary_key=True, autoincrement=True)
    prediction_id = Column(Integer, ForeignKey('predictions.prediction_id'))
    actual_numbers = Column(Text, nullable=False)
    evaluated_at = Column(String, nullable=False)
    best_match = Column(Integer)
    total_matches = Column(Integer)
    prize_value = Column(Float)
    ticket_matches = Column(Text)

    prediction = relationship("Prediction", back_populates="results")


class PlayedTicket(Base):
    __tablename__ = 'played_tickets'

    play_id = Column(Integer, primary_key=True, autoincrement=True)
    prediction_id = Column(Integer, ForeignKey('predictions.prediction_id'))
    ticket_numbers = Column(Text, nullable=False)
    played_at = Column(String, nullable=False)
    draw_date = Column(String, nullable=False)


class AdaptiveWeight(Base):
    __tablename__ = 'adaptive_weights'

    weight_id = Column(Integer, primary_key=True, autoincrement=True)
    updated_at = Column(String, nullable=False)
    strategy_name = Column(String, nullable=False)
    weight_type = Column(String, nullable=False)
    weight_value = Column(Float, nullable=False)
    performance_score = Column(Float)
    n_observations = Column(Integer, default=0)


class FairnessTest(Base):
    """Store statistical fairness test results"""
    __tablename__ = 'fairness_tests'

    test_id = Column(Integer, primary_key=True, autoincrement=True)
    tested_at = Column(String, nullable=False)
    test_name = Column(String, nullable=False)
    statistic_value = Column(Float)
    p_value = Column(Float)
    conclusion = Column(String)
    n_draws_tested = Column(Integer)
    details = Column(Text)


class CoverageAnalysis(Base):
    """Store portfolio coverage analysis"""
    __tablename__ = 'coverage_analyses'

    analysis_id = Column(Integer, primary_key=True, autoincrement=True)
    prediction_id = Column(Integer, ForeignKey('predictions.prediction_id'))
    analyzed_at = Column(String, nullable=False)
    pair_coverage = Column(Float)
    triple_coverage = Column(Float)
    number_coverage = Column(Float)
    avg_overlap = Column(Float)
    diversity_score = Column(Float)
    expected_3plus_prob = Column(Float)


# Database engine
engine = create_engine(f'sqlite:///{DB_PATH}', echo=False)
SessionLocal = sessionmaker(bind=engine)
_DB_BOOTSTRAPPED = False


def _load_draws_from_csv():
    """Read draws from configured CSV (header-aware)."""
    df = pd.read_csv(CSV_DRAWS_PATH)

    num_cols_upper = [f"NUM{i}" for i in range(1, NUMBERS_PER_DRAW + 1)]
    upper_map = {c.upper(): c for c in df.columns}
    if all(c in upper_map for c in num_cols_upper):
        numbers = df[[upper_map[c] for c in num_cols_upper]].copy()
    else:
        numbers = df.iloc[:, :NUMBERS_PER_DRAW].copy()

    numbers = numbers.apply(pd.to_numeric, errors="coerce").dropna().astype(int)

    date_col = None
    for candidate in ("draw_date", "date", "datum", "Date", "Datum"):
        if candidate in df.columns:
            date_col = candidate
            break
    if date_col is None:
        draw_dates = pd.Series(
            [f"draw_{i+1:05d}" for i in range(len(numbers))],
            index=numbers.index
        )
    else:
        draw_dates = df.loc[numbers.index, date_col].astype(str)

    round_col = None
    for candidate in ("round_number", "kolo", "Kolo", "round", "Round"):
        if candidate in df.columns:
            round_col = candidate
            break
    if round_col is None:
        rounds = pd.Series([None] * len(numbers), index=numbers.index)
    else:
        rounds = pd.to_numeric(df.loc[numbers.index, round_col], errors="coerce")

    data = []
    for idx in numbers.index:
        row_nums = [int(numbers.loc[idx].iloc[i]) for i in range(NUMBERS_PER_DRAW)]
        kolo_val = rounds.loc[idx]
        data.append(
            {
                "draw_date": str(draw_dates.loc[idx]),
                "round_number": None if pd.isna(kolo_val) else int(kolo_val),
                "numbers": row_nums,
            }
        )
    return data


def _sync_draws_table_from_csv():
    """Sync draws table from CSV source (scraper not required)."""
    session = SessionLocal()
    try:
        rows = _load_draws_from_csv()
        session.query(Draw).delete()
        for row in rows:
            nums = row["numbers"]
            session.add(
                Draw(
                    draw_date=row["draw_date"],
                    round_number=row["round_number"],
                    n1=nums[0],
                    n2=nums[1],
                    n3=nums[2],
                    n4=nums[3],
                    n5=nums[4],
                    n6=nums[5],
                    n7=nums[6],
                )
            )
        session.commit()
        logger.info(f"Draws synced from CSV: {len(rows)}")
    except Exception as e:
        session.rollback()
        logger.error(f"Failed syncing draws from CSV: {e}")
        raise
    finally:
        session.close()


def _bootstrap_db_once():
    global _DB_BOOTSTRAPPED
    if _DB_BOOTSTRAPPED:
        return
    Base.metadata.create_all(engine)
    _sync_draws_table_from_csv()
    _DB_BOOTSTRAPPED = True


def init_db():
    """Initialize database tables"""
    try:
        _bootstrap_db_once()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


def get_session():
    """Get database session"""
    _bootstrap_db_once()
    return SessionLocal()