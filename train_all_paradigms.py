"""
Master AI Training Orchestrator - Benchmark All AI Models
Runs:
  1. Classification AI (Triple Barrier Path-Dependent)
  2. Continuous Regression AI (Unconstrained Return Vectors)
  3. Deep Reinforcement Learning (PPO Agent)
  4. Generative Market GPT (1,000 Path Simulation)

Run: python train_all_paradigms.py
"""

import time
import os
import sys
import logging
from datetime import datetime

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/master_training.log", encoding="utf-8")
    ]
)
logger = logging.getLogger("MasterTrainingOrchestrator")


def main():
    os.makedirs("smartcrypto_ai_models", exist_ok=True)

    start_time_all = time.time()
    results = {}

    print("=" * 80)
    print("🌙 MASTER AI TRAINING BENCHMARK - SIDE-BY-SIDE RUNNER")
    print(f"   Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 1. Classification AI (Triple Barrier)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n" + "━" * 80)
    print("🚀 [1/4] EXECUTING: Triple Barrier Classification AI")
    print("━" * 80)
    t0 = time.time()
    try:
        from smartcrypto_ai_models import train_candle_ai
        train_candle_ai.main()
        results["1. Classification AI"] = f"✅ SUCCESS ({(time.time() - t0)/60:.1f} min)"
    except Exception as e:
        logger.error(f"❌ Classification AI failed: {e}", exc_info=True)
        results["1. Classification AI"] = f"❌ FAILED ({e})"

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 2. Continuous Return Regression AI
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n" + "━" * 80)
    print("🚀 [2/4] EXECUTING: Unconstrained Continuous Return Regression AI")
    print("━" * 80)
    t0 = time.time()
    try:
        from smartcrypto_ai_models import train_regression_ai
        train_regression_ai.main()
        results["2. Continuous Regression AI"] = f"✅ SUCCESS ({(time.time() - t0)/60:.1f} min)"
    except Exception as e:
        logger.error(f"❌ Continuous Regression AI failed: {e}", exc_info=True)
        results["2. Continuous Regression AI"] = f"❌ FAILED ({e})"

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 3. Deep Reinforcement Learning (PPO Agent)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n" + "━" * 80)
    print("🚀 [3/4] EXECUTING: Deep Reinforcement Learning (PPO Agent)")
    print("━" * 80)
    t0 = time.time()
    try:
        from smartcrypto_ai_models import train_rl_ppo
        train_rl_ppo.main()
        results["3. PPO Agent"] = f"✅ SUCCESS ({(time.time() - t0)/60:.1f} min)"
    except Exception as e:
        logger.error(f"❌ PPO Agent failed: {e}", exc_info=True)
        results["3. PPO Agent"] = f"❌ FAILED ({e})"

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 4. Generative Market GPT World Model
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n" + "━" * 80)
    print("🚀 [4/4] EXECUTING: Generative Market GPT World Model")
    print("━" * 80)
    t0 = time.time()
    try:
        from smartcrypto_ai_models import generative_market_gpt
        generative_market_gpt.main()
        results["4. Market GPT"] = f"✅ SUCCESS ({(time.time() - t0)/60:.1f} min)"
    except Exception as e:
        logger.error(f"❌ Market GPT failed: {e}", exc_info=True)
        results["4. Market GPT"] = f"❌ FAILED ({e})"

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # SUMMARY REPORT
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    total_duration_min = (time.time() - start_time_all) / 60.0

    print("\n" + "=" * 80)
    print("🎉 MASTER AI TRAINING BENCHMARK COMPLETED!")
    print(f"   Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Total Duration: {total_duration_min:.1f} minutes")
    print("=" * 80)
    print("📊 BENCHMARK SUMMARY RESULTS:")
    for paradigm, status in results.items():
        print(f"   • {paradigm:<35}: {status}")
    print("=" * 80)
    print("📁 Saved Models Location: /smartcrypto_ai_models/")
    print("=" * 80)


if __name__ == "__main__":
    main()