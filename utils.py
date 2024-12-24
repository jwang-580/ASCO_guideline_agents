
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import os
import time
import base64
import anthropic
import glob

# download all the pdfs

def download_and_rename_pdf(pdf_url, new_filename):
    # Create pdfs directory if it doesn't exist
    if not os.path.exists('pdfs'):
        os.makedirs('pdfs')

    if os.path.exists(os.path.join('pdfs', f'{new_filename}.pdf')):
        print(f"File already exists for {new_filename}")
        return
    
    # Set up Chrome options
    chrome_options = Options()
    prefs = {
        "download.default_directory": os.path.abspath("pdfs"),
        "download.prompt_for_download": False,
        "plugins.always_open_pdf_externally": True
    }
    chrome_options.add_experimental_option("prefs", prefs)
    driver = webdriver.Chrome(options=chrome_options)
    
    try:
        before_download = set(os.listdir('pdfs'))
        driver.get(pdf_url)
        
        # Wait for download to complete
        time.sleep(5)
        
        after_download = set(os.listdir('pdfs'))
        
        # Find the new file
        new_files = after_download - before_download
        if new_files:
            downloaded_file = os.path.join('pdfs', list(new_files)[0])
            new_file_path = os.path.join('pdfs', f"{new_filename}.pdf")
            
            if os.path.exists(downloaded_file):
                os.rename(downloaded_file, new_file_path)
                print(f"File renamed to: {new_filename}.pdf")
        else:
            print("No new files were downloaded")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    
    finally:
        driver.quit()


# pdf read tool
def process_pdf(key: str, prompt: str) -> str:
    try:
        pdf_path = os.path.join('pdfs', f'{key}.pdf')
        if not os.path.exists(pdf_path):
            matching_files = glob.glob(os.path.join('pdfs', f'{key}_*.pdf'))
            if matching_files:
                pdf_path = matching_files[0]
            else:
                raise FileNotFoundError(f"No PDF file found for key: {key}")

        with open(pdf_path, 'rb') as file:
            pdf_data = base64.b64encode(file.read()).decode('utf-8')

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": pdf_data
                        },
                        "cache_control": {"type": "ephemeral"}
                    },
                    {
                        "type": "text",
                        "text": f"{prompt}"
                    }
                ],
            }
        ]

        client = anthropic.Anthropic()

        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=messages
        )


        # count the tokens
        response = client.beta.messages.count_tokens(
            betas=["pdfs-2024-09-25"],
            model="claude-3-5-sonnet-20241022",
            messages=messages
        )

        print ('study:', key, '\nprompt:', prompt, '\nresponse:', response.json())
        return message.content[0].text

    except FileNotFoundError as e:
        return f"Error: {str(e)}"
    except anthropic.APIError as e:
        return f"API Error: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"