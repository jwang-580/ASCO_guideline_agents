import os
import autogen
import anthropic
import base64
import time
import glob
import json
from config import ANTHROPIC_API_KEY
from utils import download_and_rename_pdf, process_pdf
from data.asco_guidelines import guideline_urls as asco_guideline_url
from data.asco_guidelines import guideline_summaries as asco_guideline_summary

os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY

config_list_claude = [
    {
        "model": "claude-3-5-sonnet-20241022",
        "api_key": os.getenv("ANTHROPIC_API_KEY"),
        "api_type": "anthropic",
    }
]

# check and download all the pdfs
for key, value in asco_guideline_url.items():
    pdf_url = value[1]
    download_and_rename_pdf(pdf_url, key)


llm_config = {"config_list": config_list_claude, "cache_seed": 45}

user_proxy = autogen.UserProxyAgent(
    name="User_proxy",
    system_message="A human admin.",
    code_execution_config={
        "last_n_messages": 2,
        "work_dir": "groupchat",
        "use_docker": False,
    },  
    human_input_mode="TERMINATE",
)

coordinator = autogen.AssistantAgent(
    name="coordinator",
    system_message=
    f'''
    You are a coordinator who can help the user find the correct corresponding ASCO guidelines.
    You have access to a dictionary whose keys are the guideline numbers and values are descriptions of the guidelines: {asco_guideline_summary}
    Based on the user's prompt, you will determine which ASCO guideline to use, 
    and then return the key as a string (e.g. breast_cancer_8) and the user prompt, and ask pdf_viewer to retrieve information from the pdf.
    If none of the ASCO guidelines are relevant, return "none" and ask User_proxy to terminate.
    ''',
    llm_config=llm_config,
)

pdf_viewer = autogen.AssistantAgent(
    name="pdf_viewer",
    system_message="You are a pdf viewer who can view the ASCO guidelines and provide feedback.",
    llm_config=llm_config,
)

# Register the Claude PDF processing tool
from autogen import register_function
register_function(
    process_pdf, 
    caller=coordinator,
    executor=pdf_viewer,
    name="process_pdf", 
    description="Retrieve information from pdf.")

groupchat = autogen.GroupChat(agents=[user_proxy, coordinator, pdf_viewer], messages=[], max_round=12)
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

user_proxy.initiate_chat(manager, message="How to give adjuvant pembro with radiation therapy for patietns with localized triple-negative breast cancer?")