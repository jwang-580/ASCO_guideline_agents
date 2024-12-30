from evaluate_answers import AnswerEvaluator
import argparse
from datetime import datetime
import os
import pandas as pd
import time

def main():
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run evaluation on a range of questions')
    parser.add_argument('--start', type=int, default=0, help='Starting index of questions (inclusive)')
    parser.add_argument('--end', type=int, default=5, help='Ending index of questions (exclusive)')
    parser.add_argument('--seed', type=int, default=42, help='Cache seed for Claude chat')
    parser.add_argument('--interval', type=int, default=90, help='Time between evaluations in seconds (default: 90)')
    args = parser.parse_args()

    # Initialize evaluator
    print(f"Initializing evaluator with seed {args.seed}")
    evaluator = AnswerEvaluator(cache_seed=args.seed)
    
    all_results = []
    current_idx = args.start
    
    while current_idx < args.end:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{timestamp}] Evaluating question {current_idx}")
        
        # Run evaluation for single question
        results = evaluator.run_evaluation(start_idx=current_idx, end_idx=current_idx + 1)
        all_results.extend(results)
        
        # Print interim results
        total = len(all_results)
        correct_guidelines = sum(1 for r in all_results if r.get('guideline_match', False))
        correct_answers = sum(1 for r in all_results 
                            if r.get('answer_evaluation') 
                            and r['answer_evaluation'].startswith('YES'))
        
        print("\nCurrent Progress:")
        print("-" * 50)
        print(f"Questions evaluated: {total}")
        if total > 0:
            print(f"Correct guidelines: {correct_guidelines}/{total} ({correct_guidelines/total*100:.1f}%)")
            print(f"Correct answers: {correct_answers}/{total} ({correct_answers/total*100:.1f}%)")
        
        current_idx += 1
        
        # Wait before next evaluation (unless it's the last one)
        if current_idx < args.end:
            print(f"\nWaiting {args.interval} seconds before next evaluation...")
            time.sleep(args.interval)
    
    # Save final results to CSV
    results_df = pd.DataFrame(all_results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f'results/evaluation_results_{timestamp}.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"\nFinal results saved to: {csv_path}")

if __name__ == "__main__":
    main() 