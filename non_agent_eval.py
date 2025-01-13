# evaluate the answers without the agent workflow
# evalute both Claude 3.5 Sonnet and GPT-4o, using Claude 3.5 Sonnet as the judge 

from dotenv import load_dotenv
from openai import OpenAI
import anthropic
import pandas as pd
import os
import argparse
from datetime import datetime
load_dotenv()
oai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
claude_client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

class NonAgentEval:
    def __init__(self, qa_df):
        self.qa_df = qa_df

    def generate_answer(self, start_idx, end_idx):
        """
        Generate answers for a range of questions by both Claude 3.5 Sonnet and GPT-4o
        """
        selected_qa = self.qa_df.iloc[start_idx:end_idx]
        oai_answers = []
        claude_answers = []

        for _, row in selected_qa.iterrows():
            question = row['Question']
            oai_response = oai_client.chat.completions.create(
                model="gpt-4o-2024-11-20",
                messages=[{"role": "user", 
                           "content": f"please provide a short and concise answer to the following question: {question}"}],
                temperature=0.0,
            )
            oai_answers.append(oai_response.choices[0].message.content)

            claude_response = claude_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=500,
                messages=[{"role": "user", 
                           "content": f"please provide a short and concise answer to the following question: {question}"}],
                temperature=0.0,
            )
            claude_answers.append(claude_response.content[0].text)

        return oai_answers, claude_answers


    def evaluate_single_answer(self, question, generated_answer, expected_answer):
        """
        Send the comparison task to Claude to evaluate if the answers match in meaning
        """
        prompt = f"""
        Compare these two answers to the question: "{question}"
        
        Generated answer: {generated_answer}
        Expected answer: {expected_answer}
        
        Do these answers convey the same key information and recommendations? 
        Respond with only 'YES' or 'NO'.
        """
        
        response = claude_client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens = 300,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Extract just the text from the response
        return response.content[0].text if isinstance(response.content, list) else response.content
    
    def evaluate_batch(self, start_idx, end_idx):
        """
        Evaluate a batch of questions and save results
        """
        oai_answers, claude_answers = self.generate_answer(start_idx, end_idx)
        selected_qa = self.qa_df.iloc[start_idx:end_idx]
        
        results = []
        for i, (_, row) in enumerate(selected_qa.iterrows()):
            question = row['Question']
            expected = row['Answer']
            
            oai_match = self.evaluate_single_answer(question, oai_answers[i], expected)
            claude_match = self.evaluate_single_answer(question, claude_answers[i], expected)
            
            results.append({
                'question': question,
                'expected_answer': expected,
                'oai_answer': oai_answers[i],
                'claude_answer': claude_answers[i],
                'oai_match': oai_match,
                'claude_match': claude_match
            })
        
        results_df = pd.DataFrame(results)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        results_df.to_csv(f'results/non_agent_evaluation_results_{timestamp}.csv', index=False)
        return results_df


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Evaluate AI model answers')
    parser.add_argument('--start', type=int, default=0, help='Starting index for evaluation')
    parser.add_argument('--end', type=int, help='Ending index for evaluation (defaults to all questions)')
    
    args = parser.parse_args()
    
    # Load the QA dataset
    qa_df = pd.read_csv('data/q_a.csv')
    
    # Initialize the evaluator
    evaluator = NonAgentEval(qa_df)
    
    # Set end index to length of dataset if not specified
    end_idx = args.end if args.end is not None else len(qa_df)
    
    try:
        # Process questions within specified range
        print(f"Processing questions from index {args.start} to {end_idx}...")
        results = evaluator.evaluate_batch(args.start, end_idx)
        
        # Print final results
        print(f"\nFinal results:")
        print(f"GPT-4 matches: {(results['oai_match'] == 'YES').sum()}/{len(results)}")
        print(f"Claude matches: {(results['claude_match'] == 'YES').sum()}/{len(results)}")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    
    print("\nEvaluation complete. Results saved to results/non_agent_evaluation_results.csv")