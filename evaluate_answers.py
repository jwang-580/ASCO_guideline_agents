import pandas as pd
import anthropic
import os
from config import ANTHROPIC_API_KEY
import claude_autogen
import autogen
import glob
from claude_autogen import ClaudeChat

class AnswerEvaluator:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self.qa_df = pd.read_csv('data/q_a.csv')
        
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
        
        response = self.client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=300,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content
    
    def extract_guideline_from_chat(self, chat_messages):
        """
        Extract the guideline number from the chat messages
        """
        for msg in chat_messages:
            if msg.get("name") == "coordinator":
                content = msg.get("content", "")
                if "key" in content and "_cancer_" in content:
                    return content.split('"key": "')[1].split('"')[0]
        return None

    def run_evaluation(self, num_questions=5):
        """
        Run evaluation on a subset of questions
        """
        results = []
        
        # Take a sample of questions
        sample_qa = self.qa_df.sample(n=num_questions)
        
        for _, row in sample_qa.iterrows():
            question = row['Question']
            expected_answer = row['Answer']
            expected_guideline = row['Guideline']
            
            print(f"\nEvaluating question: {question}")
            
            claude_chat = ClaudeChat()
            chat_messages = claude_chat.chat(question)
            
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
                'expected_guideline': expected_guideline,
                'generated_guideline': generated_guideline,
                'guideline_match': expected_guideline == generated_guideline,
                'answer_evaluation': evaluation
            })
            
        return results

def main():
    evaluator = AnswerEvaluator()
    results = evaluator.run_evaluation(num_questions=5)
    
    # Print results
    for i, result in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"Question: {result['question']}")
        print(f"Guideline Match: {result['guideline_match']}")
        print(f"Expected Guideline: {result['expected_guideline']}")
        print(f"Generated Guideline: {result['generated_guideline']}")
        print(f"Answer Evaluation: {result['answer_evaluation']}")

if __name__ == "__main__":
    main() 