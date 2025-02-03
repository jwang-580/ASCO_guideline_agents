import pandas as pd
from openai import OpenAI
import os
from config import OPENAI_API_KEY
import re
from datetime import datetime
from claude_autogen import ClaudeChat

class AnswerEvaluator:
    def __init__(self, cache_seed):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.qa_df = pd.read_csv('data/q_a.csv')
        self.cache_seed = cache_seed

    def evaluate_single_answer(self, question, generated_answer, expected_answer):
        """
        Send the comparison task to GPT-4o to evaluate if the answers match in meaning
        """
        prompt = f"""
        Compare these two answers to the question: "{question}"
        
        Generated answer: {generated_answer}
        Expected answer: {expected_answer}
        
        Do these answers convey the same key information and recommendations? 
        Respond with only 'YES' or 'NO'.
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        
        # Extract just the text from the response
        return response.choices[0].message.content
    
    def extract_guideline_from_chat(self, chat_messages):
        """
        Extract the guideline number from the chat messages
        """
        for msg in chat_messages:
            if msg.get("name") == "coordinator":
                content = msg.get("content", "")
                pattern = r'[a-z]+_cancer_\d+'
                match = re.search(pattern, content)
                if match:
                    return match.group(0)
        return None

    def run_evaluation(self, start_idx=0, end_idx=5):
        """
        Run evaluation on a range of questions from start_idx to end_idx
        
        Args:
            start_idx (int): Starting index of questions (inclusive)
            end_idx (int): Ending index of questions (exclusive)
        """
        results = []
        
        # Get specific range of questions
        selected_qa = self.qa_df.iloc[start_idx:end_idx]
        
        for _, row in selected_qa.iterrows():
            question = row['Question']
            expected_answer = row['Answer']
            expected_guideline = row['Guideline']
            
            print(f"\nEvaluating question: {question}")
            
            claude_chat = ClaudeChat(cache_seed=self.cache_seed)
            chat_result = claude_chat.chat(question)
            
            # Access the messages from the ChatResult object
            chat_messages = chat_result.chat_history
            
            # Extract the generated answer and guideline
            generated_answer = None
            for msg in reversed(chat_messages):
                if msg.get("name") == "reviewer":
                    generated_answer = msg.get("content")
                    break
                    
            generated_guideline = self.extract_guideline_from_chat(chat_messages)
            
            # Evaluate the answer
            if generated_answer:
                evaluation = self.evaluate_single_answer(
                    question, 
                    generated_answer, 
                    expected_answer
                )
            else:
                evaluation = "NO - No answer generated"
            
            # Store results
            results.append({
                'question': question,
                'expected_answer': expected_answer,
                'generated_answer': generated_answer,
                'expected_guideline': expected_guideline,
                'generated_guideline': generated_guideline,
                'guideline_match': expected_guideline == generated_guideline,
                'answer_correct': evaluation
            })
            
        return results

def main():
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    evaluator = AnswerEvaluator(cache_seed=42)
    results = evaluator.run_evaluation(start_idx=5, end_idx=6)
    
    # Print results
    for i, result in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"Question: {result['question']}")
        print(f"Guideline Match: {result['guideline_match']}")
        print(f"Expected Guideline: {result['expected_guideline']}")
        print(f"Generated Guideline: {result['generated_guideline']}")
        print(f"Expected Answer: {result['expected_answer']}")
        print(f"Generated Answer: {result['generated_answer']}")
        print(f"Answer Correct: {result['answer_correct']}")

    # save results to csv
    results_df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df.to_csv(f'results/evaluation_results_{timestamp}.csv', index=False)

if __name__ == "__main__":
    main() 