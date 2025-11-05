"""
Leave-One-Out Evaluation: Tests the multi-agent framework by excluding the correct guideline summary
This evaluation measures how well the system performs when the correct guideline is masked from the coordinator.
"""

from evaluate_answers import AnswerEvaluator
import argparse
from datetime import datetime
import os
import pandas as pd
import random
from data.asco_guidelines import guideline_summaries


class LeaveOneOutEvaluator(AnswerEvaluator):
    """
    Extends AnswerEvaluator to test performance when correct guideline summary is excluded
    """
    
    def __init__(self, cache_seed):
        super().__init__(cache_seed)
        
    def create_masked_summaries(self, guideline_to_exclude):
        """
        Create a modified guideline_summaries dictionary that excludes the specified guideline
        
        Args:
            guideline_to_exclude (str): The guideline key to exclude (e.g., 'breast_cancer_1')
        
        Returns:
            dict: Modified guideline summaries without the excluded guideline
        """
        masked_summaries = {k: v for k, v in guideline_summaries.items() if k != guideline_to_exclude}
        print(f"Masked guideline: {guideline_to_exclude}")
        print(f"Available guidelines: {len(masked_summaries)} (original: {len(guideline_summaries)})")
        return masked_summaries
    
    def run_leave_one_out_evaluation(self, question_indices):
        """
        Run evaluation with correct guideline summaries masked
        
        Args:
            question_indices (list): List of question indices to evaluate
        
        Returns:
            list: Results with evaluation metrics
        """
        results = []
        
        for idx in question_indices:
            row = self.qa_df.iloc[idx]
            question = row['Question']
            expected_answer = row['Answer']
            expected_guideline = row['Guideline']
            
            print(f"\n{'='*70}")
            print(f"Evaluating question {idx}: {question[:80]}...")
            print(f"Expected guideline (masked): {expected_guideline}")
            print(f"{'='*70}")
            
            # Create masked summaries excluding the correct guideline
            masked_summaries = self.create_masked_summaries(expected_guideline)
            
            try:
                # Import here to create a fresh instance
                from claude_autogen import ClaudeChat
                
                # Run evaluation with masked guideline summaries
                claude_chat = ClaudeChat(
                    cache_seed=self.cache_seed,
                    custom_guideline_summaries=masked_summaries
                )
                chat_result = claude_chat.chat(question)
                
                # Access chat messages
                chat_messages = chat_result.chat_history
                
                # Extract final answer from reviewer
                generated_answer = None
                for msg in reversed(chat_messages):
                    if msg.get("name") == "reviewer":
                        generated_answer = msg.get("content")
                        break
                
                # Extract guideline chosen by coordinator
                generated_guideline = self.extract_guideline_from_chat(chat_messages)
                
                # Evaluate answer correctness
                if generated_answer:
                    evaluation = self.evaluate_single_answer(question, generated_answer, expected_answer)
                else:
                    evaluation = "NO"
                
                # Store results
                results.append({
                    'question_index': idx,
                    'question': question,
                    'expected_answer': expected_answer,
                    'generated_answer': generated_answer,
                    'expected_guideline': expected_guideline,
                    'generated_guideline': generated_guideline,
                    'guideline_match': expected_guideline == generated_guideline,
                    'answer_correct': evaluation
                })
                
            except Exception as e:
                print(f"Error during evaluation: {str(e)}")
                results.append({
                    'question_index': idx,
                    'question': question,
                    'expected_answer': expected_answer,
                    'generated_answer': None,
                    'expected_guideline': expected_guideline,
                    'generated_guideline': None,
                    'guideline_match': False,
                    'answer_correct': f"ERROR - {str(e)}"
                })
        
        return results


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Leave-one-out evaluation: Test with correct guideline summary masked from coordinator'
    )
    parser.add_argument(
        '--num_questions', 
        type=int, 
        default=10, 
        help='Number of random questions to evaluate (default: 10)'
    )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42, 
        help='Random seed for question selection (default: 42)'
    )
    parser.add_argument(
        '--cache_seed', 
        type=int, 
        default=42, 
        help='Cache seed for model chat (default: 42)'
    )
    parser.add_argument(
        '--specific_indices',
        type=str,
        default=None,
        help='Comma-separated list of specific question indices to evaluate (e.g., "0,5,10")'
    )
    args = parser.parse_args()

    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Initialize evaluator
    print(f"Initializing leave-one-out evaluator with cache seed {args.cache_seed}")
    evaluator = LeaveOneOutEvaluator(cache_seed=args.cache_seed)
    
    # Select questions
    if args.specific_indices:
        # Use specific indices provided by user
        question_indices = [int(idx.strip()) for idx in args.specific_indices.split(',')]
        print(f"Using specific question indices: {question_indices}")
    else:
        # Randomly select questions
        random.seed(args.seed)
        total_questions = len(evaluator.qa_df)
        question_indices = random.sample(range(total_questions), min(args.num_questions, total_questions))
        question_indices.sort()
        print(f"Randomly selected {len(question_indices)} questions (seed={args.seed}): {question_indices}")
    
    # Run leave-one-out evaluation
    print("\n" + "="*70)
    print("LEAVE-ONE-OUT EVALUATION: Testing with correct guideline summary masked")
    print("="*70)
    
    results = evaluator.run_leave_one_out_evaluation(question_indices)
    
    # Calculate statistics
    total = len(results)
    correct_guidelines = sum(1 for r in results if r.get('guideline_match', False))
    correct_answers = sum(1 for r in results 
                         if r.get('answer_correct') 
                         and isinstance(r['answer_correct'], str)
                         and r['answer_correct'].startswith('YES'))
    
    # Print summary
    print("\n" + "="*70)
    print("LEAVE-ONE-OUT EVALUATION SUMMARY")
    print("="*70)
    print(f"Total questions evaluated: {total}")
    print(f"Correct guidelines (should be 0): {correct_guidelines}/{total} ({correct_guidelines/total*100:.1f}%)")
    print(f"Correct answers: {correct_answers}/{total} ({correct_answers/total*100:.1f}%)")
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f'results/leave_one_out_evaluation_results_{timestamp}.csv'
    results_df.to_csv(csv_path, index=False)
    
    print(f"\nResults saved to: {csv_path}")
    print("\nDetailed Results:")
    print("-" * 70)
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Question {result['question_index']}: {result['question'][:60]}...")
        print(f"   Expected guideline: {result['expected_guideline']}")
        print(f"   Generated guideline: {result['generated_guideline']}")
        print(f"   Guideline match: {result['guideline_match']}")
        print(f"   Answer correct: {result['answer_correct']}")


if __name__ == "__main__":
    main()

