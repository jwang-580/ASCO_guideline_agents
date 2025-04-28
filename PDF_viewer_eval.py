from config import OPENAI_API_KEY
from config import ANTHROPIC_API_KEY
import os
import pandas as pd
import openai
import anthropic
from llama_index.core import SimpleDirectoryReader
from llama_index.core import GPTVectorStoreIndex
from datetime import datetime
from typing import Dict

def create_client(model_choice: str = "gpt-4o"):
    """
    Create and return an API client for either OpenAI or Anthropic.
    """
    MODEL_MAPPING = {
        "gpt-4o": "gpt-4o-2024-11-20",
        "claude": "claude-3-7-sonnet-20250219"
    }
    
    if model_choice not in MODEL_MAPPING:
        raise ValueError("model_choice must be either 'gpt-4o' or 'claude'")
        
    if model_choice == "gpt-4o":
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
    else:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    
    return client, MODEL_MAPPING[model_choice]

def query_llm(client, model: str, prompt: str) -> str:
    """
    Query the LLM with a given prompt and return the response text.
    """
    if isinstance(client, openai.OpenAI):
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    else:  # Anthropic
        message = client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content

def build_index_for_pdf(pdf_path: str, chunk_size: int = 1024) -> GPTVectorStoreIndex:
    """
    Build a GPTVectorStoreIndex for a single PDF file.
    """
    documents = SimpleDirectoryReader(input_files=[pdf_path]).load_data()
    index = GPTVectorStoreIndex.from_documents(documents)
    return index

def build_all_indices(pdf_folder: str, chunk_size: int = 1024) -> Dict[str, GPTVectorStoreIndex]:
    """
    Build indices for all PDF files in the folder and return a dictionary mapping filenames to indices.
    """
    indices = {}
    for filename in os.listdir(pdf_folder):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(pdf_folder, filename)
            guideline_name = filename[:-4]  # Remove .pdf extension
            print(f"Building index for {filename}...")
            indices[guideline_name] = build_index_for_pdf(pdf_path, chunk_size)
    return indices

def query_index(index: GPTVectorStoreIndex, question: str) -> str:
    """
    Query the LlamaIndex with a given question and return the response text.
    """
    query_engine = index.as_query_engine()
    response = query_engine.query(question)
    return str(response)

def evaluate_answer(client, model: str, question: str, generated_answer: str, expected_answer: str) -> str:
    """
    Use GPT-4 to evaluate if the generated answer matches the expected answer in meaning.
    """
    prompt = f"""
    Compare these two answers to the question: "{question}"
    
    Generated answer: {generated_answer}
    Expected answer: {expected_answer}
    
    Do these answers convey the same key information and recommendations? 
    Respond with only 'YES' or 'NO'.
    """
    
    return query_llm(client, model, prompt)

def main(
    pdf_folder: str = "pdfs",
    csv_path: str = "data/q_a.csv",
    start_idx: int = 0,
    end_idx: int = 9,
    model_choice: str = "gpt-4o",
    chunk_size: int = 1024,
    output_csv: str = None
):
    """
    Build separate indices for each PDF and evaluate questions using the corresponding PDF.
    """
    # Set default output_csv path if none provided
    if output_csv is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_csv = f"results/q_a_answered_PDF_{model_choice}_{timestamp}.csv"

    # Create API clients
    client, model = create_client(model_choice)
    eval_client, eval_model = create_client("gpt-4o")  # Always use GPT-4 for evaluation

    # Build indices for all PDFs
    print("Building indices for all PDFs...")
    indices = build_all_indices(pdf_folder, chunk_size)
    print("All indices built successfully.\n")

    # Read CSV of questions
    df = pd.read_csv(csv_path)
    
    required_columns = ['Question', 'Answer', 'Guideline']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"CSV file must contain a column named '{col}'")
        
    # Add columns for generated answer and evaluation
    df['Generated_answer'] = ""
    df['Matches_Expected'] = ""

    # Process questions
    end_idx = min(end_idx, len(df) - 1)
    for i in range(start_idx, end_idx + 1):
        question = df.loc[i, 'Question']
        guideline = df.loc[i, 'Guideline']
        
        print(f"\nProcessing row {i}:")
        print(f"Question: {question}")
        print(f"Using guideline: {guideline}")
        
        if guideline not in indices:
            print(f"Warning: No index found for guideline {guideline}")
            continue
            
        # Get context from the specific index
        context = query_index(indices[guideline], question)
        print(f"\nRetrieved Context:\n{context}\n")
        
        # Create prompt with context
        prompt = f"""Based on the following context, please answer the question. Please provide a short and concise answer. If the answer is not found in the context, please say so.
                    Context: {context}
                    Question: {question}
                    Answer:"""
        
        # Query LLM with context and question
        answer = query_llm(client, model, prompt)
        df.loc[i, "Generated_answer"] = answer
        print(f"Generated Answer: {answer}\n")

        # Evaluate the answer if there's an expected answer
        if df.loc[i, "Answer"]:
            evaluation = evaluate_answer(
                eval_client, 
                eval_model,
                question, 
                answer, 
                df.loc[i, "Answer"]
            )
            df.loc[i, "Matches_Expected"] = evaluation
            print(f"Matches Expected Answer: {evaluation}\n")

        print("-" * 80)

    # Save results
    df.to_csv(output_csv, index=False)
    print(f"\nAnswers written to {output_csv}")
    
    # Print summary statistics
    if 'Matches_Expected' in df.columns:
        matches = (df['Matches_Expected'] == 'YES').sum()
        total_evaluated = df['Matches_Expected'].notna().sum()
        print(f"\nEvaluation Summary:")
        print(f"Matches: {matches}/{total_evaluated} ({(matches/total_evaluated*100):.1f}% match rate)")

if __name__ == "__main__":
    main(
        pdf_folder="pdfs",
        csv_path="data/q_a.csv",
        start_idx=0,
        end_idx=99,
        model_choice="gpt-4o",
        chunk_size=1024
    )