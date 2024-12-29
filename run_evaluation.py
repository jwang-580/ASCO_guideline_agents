from evaluate_answers import AnswerEvaluator
import argparse
from datetime import datetime
import os
import pandas as pd

def main():
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run evaluation on a range of questions')
    parser.add_argument('--start', type=int, default=0, help='Starting index of questions (inclusive)')
    parser.add_argument('--end', type=int, default=5, help='Ending index of questions (exclusive)')
    parser.add_argument('--seed', type=int, default=42, help='Cache seed for Claude chat')
    args = parser.parse_args()

    # Initialize evaluator
    print(f"Initializing evaluator with seed {args.seed}")
    evaluator = AnswerEvaluator(cache_seed=args.seed)
    
    # Run evaluation
    print(f"\nEvaluating questions from index {args.start} to {args.end}")
    results = evaluator.run_evaluation(start_idx=args.start, end_idx=args.end)
    
    # Print results summary
    print("\nEvaluation Summary:")
    print("-" * 50)
    
    total = len(results) if results else 0
    if total == 0:
        print("No results to evaluate")
        return

    correct_guidelines = sum(1 for r in results if r.get('guideline_match', False))
    
    correct_answers = sum(1 for r in results 
                         if r.get('answer_evaluation') 
                         and r['answer_evaluation'].startswith('YES'))
    
    print(f"Total questions evaluated: {total}")
    if total > 0:
        print(f"Correct guidelines: {correct_guidelines}/{total} ({correct_guidelines/total*100:.1f}%)")
        print(f"Correct answers: {correct_answers}/{total} ({correct_answers/total*100:.1f}%)")

    # Save results to CSV
    results_df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f'results/evaluation_results_{timestamp}.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

if __name__ == "__main__":
    main() 