# evaluate the answers without the agent workflow
# evalute both Claude 3.5 Sonnet and GPT-4o, using Claude 3.5 Sonnet as the judge 

from dotenv import load_dotenv
from openai import OpenAI
from google import genai
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
import anthropic
import pandas as pd
import os
import argparse
from datetime import datetime


load_dotenv()
oai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
claude_client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
gemini_client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
azure_client = ChatCompletionsClient(
    endpoint="https://aistudioaiservices636633355478.services.ai.azure.com/models",
    credential=AzureKeyCredential(os.getenv('AZURE_API_KEY'))
)

class NonAgentEval:
    def __init__(self, qa_df):
        self.qa_df = qa_df

    def generate_answer(self, start_idx, end_idx, model_name):
        """
        Generate answers for a range of questions using the specified model
        """
        selected_qa = self.qa_df.iloc[start_idx:end_idx]
        answers = []

        for _, row in selected_qa.iterrows():
            question = row['Question']
            answer = None

            if model_name == "gpt-4o":
                response = oai_client.chat.completions.create(
                    model="gpt-4o-2024-11-20",
                    messages=[{"role": "user", 
                            "content": f"please provide a short and concise answer to the following question: {question}"}],
                    temperature=0.0,
                )
                answer = response.choices[0].message.content

            elif model_name == "claude-3-7":
                response = claude_client.messages.create(
                    model="claude-3-7-sonnet-20250219",
                    max_tokens=500,
                    messages=[{"role": "user", 
                            "content": f"please provide a short and concise answer to the following question: {question}"}],
                    temperature=0.0,
                )
                answer = response.content[0].text

            elif model_name == "gemini-2.5-flash":
                response = gemini_client.models.generate_content(
                    model="gemini-2.5-flash-preview-04-17",
                    contents=f"please provide a short and concise answer to the following question: {question}"
                )
                answer = response.text

            elif model_name == "DeepSeek-R1":
                response = azure_client.complete(
                    messages=[
                        SystemMessage(content="You are a helpful assistant."),
                        UserMessage(content=f"please provide a short and concise answer to the following question: {question}")
                    ],
                    max_tokens=2048,
                    model="DeepSeek-R1"
                )
                answer_raw = response.choices[0].message.content
                answer = answer_raw.split("</think>")[1].strip()
            
            else:
                raise ValueError(f"Unsupported model: {model_name}")

            answers.append(answer)
            
        return answers


    def evaluate_single_answer(self, question, generated_answer, expected_answer):
        """
        Send the comparison task to Claude to evaluate if the answers match in meaning
        """
        prompt = f"""
        Compare these two answers to the question: "{question}"
        
        Generated answer: {generated_answer}
        Expected answer: {expected_answer}
        
        Does the generated answer contain the same key information as the expected answer? 
        Respond with only 'YES' or 'NO'.
        """
        
        response = oai_client.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        
        # Extract just the text from the response
        return response.choices[0].message.content
    
    def evaluate_batch(self, start_idx, end_idx, model_name):
        """
        Evaluate a batch of questions and save results
        """
        model_answers = self.generate_answer(start_idx, end_idx, model_name)
        selected_qa = self.qa_df.iloc[start_idx:end_idx]
        
        results = []
        for i, (_, row) in enumerate(selected_qa.iterrows()):
            question = row['Question']
            expected = row['Answer']
            
            match = self.evaluate_single_answer(question, model_answers[i], expected)
            
            results.append({
                'question': question,
                'expected_answer': expected,
                f'{model_name}_answer': model_answers[i],
                f'{model_name}_match': match
            })
        
        results_df = pd.DataFrame(results)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_file = f'results/non_agent_evaluation_results_{model_name}_{timestamp}.csv'
        results_df.to_csv(output_file, index=False)
        return results_df


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Evaluate AI model answers')
    parser.add_argument('--start', type=int, default=0, help='Starting index for evaluation')
    parser.add_argument('--end', type=int, help='Ending index for evaluation (defaults to all questions)')
    parser.add_argument('--model', type=str, default="gpt-4o", help='Model to evaluate (gpt-4o, claude-3-7, gemini-2.5-flash, or DeepSeek-R1)')
    args = parser.parse_args()
    
    # Load the QA dataset
    qa_df = pd.read_csv('data/q_a.csv')
    
    # Initialize the evaluator
    evaluator = NonAgentEval(qa_df)
    
    # Set end index to length of dataset if not specified
    end_idx = args.end if args.end is not None else len(qa_df)
    
    try:
        # Process questions within specified range
        print(f"Processing questions from index {args.start} to {end_idx} using {args.model}...")
        results = evaluator.evaluate_batch(args.start, end_idx, args.model)
        
        # Print final results
        print(f"\nFinal results:")
        print(f"{args.model} matches: {(results[f'{args.model}_match'] == 'YES').sum()}/{len(results)}")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    
    print(f"\nEvaluation complete. Results saved to results/non_agent_evaluation_results_{args.model}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv")