"""
Migration: old raw-SQL prediction data → SQLAlchemy tables.
"""
import sqlite3
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lotto_ai.config import DB_PATH, logger
from lotto_ai.core.db import get_session, init_db, Prediction, PredictionResult


def check_old_tables_exist():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    old_tables = {}
    for table in ['predictions', 'prediction_results', 'adaptive_weights']:
        try:
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            count = cur.fetchone()[0]
            old_tables[table] = count
        except sqlite3.OperationalError:
            old_tables[table] = 0
    conn.close()
    return old_tables


def get_old_table_columns(table_name):
    """Get actual column names from old table"""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    try:
        cur.execute(f"PRAGMA table_info({table_name})")
        columns = [row[1] for row in cur.fetchall()]
        return columns
    except Exception:
        return []
    finally:
        conn.close()


def migrate_old_predictions():
    init_db()
    old_counts = check_old_tables_exist()
    logger.info(f"Old table counts: {old_counts}")

    if old_counts.get('predictions', 0) == 0:
        logger.info("No old predictions to migrate")
        return 0

    # Check actual columns
    pred_columns = get_old_table_columns('predictions')
    logger.info(f"Old predictions columns: {pred_columns}")

    # Build SELECT query based on available columns
    has_metadata = 'metadata' in pred_columns
    has_model_metadata = 'model_metadata' in pred_columns

    if has_model_metadata:
        meta_col = 'model_metadata'
    elif has_metadata:
        meta_col = 'metadata'
    else:
        meta_col = None

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    session = get_session()
    migrated = 0

    try:
        if meta_col:
            cur.execute(f"""
                SELECT prediction_id, created_at, target_draw_date,
                       strategy_name, model_version, portfolio_size,
                       tickets, {meta_col}, evaluated
                FROM predictions ORDER BY prediction_id
            """)
        else:
            cur.execute("""
                SELECT prediction_id, created_at, target_draw_date,
                       strategy_name, model_version, portfolio_size,
                       tickets, NULL, evaluated
                FROM predictions ORDER BY prediction_id
            """)

        old_predictions = cur.fetchall()

        for row in old_predictions:
            old_id = row[0]

            existing = session.query(Prediction).filter_by(
                created_at=row[1],
                target_draw_date=row[2],
                strategy_name=row[3]
            ).first()

            if existing:
                continue

            pred = Prediction(
                created_at=row[1],
                target_draw_date=row[2],
                strategy_name=row[3],
                model_version=row[4],
                portfolio_size=row[5],
                tickets=row[6],
                model_metadata=row[7] or '{}',
                evaluated=bool(row[8]) if row[8] is not None else False
            )
            session.add(pred)
            session.flush()

            # Migrate results
            result_columns = get_old_table_columns('prediction_results')
            if 'prediction_id' in result_columns:
                cur.execute("""
                    SELECT actual_numbers, evaluated_at, best_match,
                           total_matches, prize_value, ticket_matches
                    FROM prediction_results
                    WHERE prediction_id = ?
                """, (old_id,))

                for result_row in cur.fetchall():
                    result = PredictionResult(
                        prediction_id=pred.prediction_id,
                        actual_numbers=result_row[0],
                        evaluated_at=result_row[1],
                        best_match=result_row[2],
                        total_matches=result_row[3],
                        prize_value=result_row[4],
                        ticket_matches=result_row[5]
                    )
                    session.add(result)

            migrated += 1

        session.commit()
        logger.info(f"Migrated {migrated} predictions")

    except Exception as e:
        session.rollback()
        logger.error(f"Migration error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        session.close()
        conn.close()

    return migrated


if __name__ == "__main__":
    print("=" * 70)
    print("PREDICTION DATA MIGRATION")
    print("=" * 70)

    counts = check_old_tables_exist()
    print(f"\nOld table data:")
    for table, count in counts.items():
        print(f"  {table}: {count} rows")

    columns = get_old_table_columns('predictions')
    print(f"\nPredictions columns: {columns}")

    if any(v > 0 for v in counts.values()):
        print("\nMigrating...")
        n = migrate_old_predictions()
        print(f"\nMigrated {n} predictions")
    else:
        print("\nNothing to migrate")

    print("=" * 70)