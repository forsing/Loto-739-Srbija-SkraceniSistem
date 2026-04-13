"""
Generate predictions and save them for tracking
"""
from datetime import datetime, timedelta
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lotto_ai.features.features import build_feature_matrix, load_draws
from lotto_ai.models.production_model import (
    generate_adaptive_portfolio,
    portfolio_statistics
)
from lotto_ai.tracking.prediction_tracker import PredictionTracker
from lotto_ai.learning.adaptive_learner import AdaptiveLearner

def get_next_draw_date():
    """Calculate next draw date (Tuesday or Friday)"""
    today = datetime.now()
    days_ahead = 0
    
    # Find next Tuesday (1) or Friday (4)
    while True:
        days_ahead += 1
        next_date = today + timedelta(days=days_ahead)
        if next_date.weekday() in [1, 4]:  # Tuesday or Friday
            return next_date.strftime('%Y-%m-%d')

def main():
    print("=" * 70)
    print("🎰 ADAPTIVE LOTTO MAX PREDICTOR")
    print("=" * 70)
    
    tracker = PredictionTracker()
    learner = AdaptiveLearner()
    
    # Step 1: Auto-evaluate any pending predictions
    print("\n📊 Step 1: Checking for pending predictions...")
    tracker.auto_evaluate_pending()
    
    # Step 2: Update adaptive weights based on performance
    print("\n🧠 Step 2: Updating adaptive weights...")
    updated_weights = learner.update_weights(strategy_name='hybrid_v1', window=20)
    
    # Step 3: Get performance stats
    print("\n📈 Step 3: Current performance:")
    perf = tracker.get_strategy_performance('hybrid_v1', window=50)
    if perf:
        print(f"   Last {perf['n_predictions']} predictions:")
        print(f"   • Avg best match: {perf['avg_best_match']:.2f}/7")
        print(f"   • Avg prize: ${perf['avg_prize_value']:.2f}")
        print(f"   • Hit rate (3+): {perf['hit_rate_3plus']:.1%}")
        print(f"   • Best ever: {perf['best_ever']}/7")
        print(f"   • Total won: ${perf['total_prize_won']:.2f}")
    else:
        print("   No historical data yet")
    
    # Step 4: Generate new prediction
    print("\n🎲 Step 4: Generating new portfolio...")
    features = build_feature_matrix()
    portfolio, weights = generate_adaptive_portfolio(
        features, 
        n_tickets=10, 
        use_adaptive=True
    )
    
    # Step 5: Save prediction for tracking
    next_draw = get_next_draw_date()
    print(f"\n💾 Step 5: Saving prediction for draw {next_draw}...")
    
    prediction_id = tracker.save_prediction(
        target_draw_date=next_draw,
        strategy_name='hybrid_v1',
        tickets=portfolio,
        model_version='2.0_adaptive',
        metadata=weights
    )
    
    # Step 6: Display results
    stats = portfolio_statistics(portfolio)
    
    print("\n" + "=" * 70)
    print("📦 PORTFOLIO STATISTICS")
    print("=" * 70)
    print(f"  Prediction ID:    {prediction_id}")
    print(f"  Target Draw:      {next_draw}")
    print(f"  Total Tickets:    {stats['total_tickets']}")
    print(f"  Unique Numbers:   {stats['unique_numbers']}/50 ({stats['coverage_pct']:.1f}%)")
    print(f"  Avg Overlap:      {stats['avg_overlap']:.2f} numbers")
    
    print("\n" + "=" * 70)
    print("🎟️  YOUR TICKETS")
    print("=" * 70)
    
    n_freq = weights['n_freq_tickets']
    
    print(f"\n📊 Frequency-Based Tickets ({n_freq}):")
    for i, ticket in enumerate(portfolio[:n_freq], 1):
        ticket_str = ' '.join(f'{n:2d}' for n in ticket)
        print(f"  Ticket {i:2d}: [{ticket_str}]")
    
    print(f"\n🎲 Random Variance Tickets ({weights['n_random_tickets']}):")
    for i, ticket in enumerate(portfolio[n_freq:], n_freq + 1):
        ticket_str = ' '.join(f'{n:2d}' for n in ticket)
        print(f"  Ticket {i:2d}: [{ticket_str}]")
    
    print("\n" + "=" * 70)
    print("📝 NEXT STEPS")
    print("=" * 70)
    print(f"""
  1. Play these tickets for draw on {next_draw}
  2. After the draw, run this script again to:
     • Evaluate your prediction
     • Update the learning model
     • Generate new optimized tickets
  
  🧠 The system will learn from each draw and adapt!
    """)
    print("=" * 70)

if __name__ == "__main__":
    main()