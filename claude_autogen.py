import os
import autogen
import anthropic
import base64
import time
import glob
import json
from autogen import register_function
from config import ANTHROPIC_API_KEY
from utils import download_and_rename_pdf, process_pdf
from data.asco_guidelines import guideline_urls as asco_guideline_url
from data.asco_guidelines import guideline_summaries as asco_guideline_summary

# check and download all the pdfs
for key, value in asco_guideline_url.items():
    pdf_url = value[1]
    download_and_rename_pdf(pdf_url, key)


class ClaudeChat:
    def __init__(self, cache_seed, custom_guideline_summaries=None):

        os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY

        # Use custom summaries if provided, otherwise use default
        guidelines_to_use = custom_guideline_summaries if custom_guideline_summaries is not None else asco_guideline_summary

        config_list_claude = [
            {
                # "model": "claude-3-5-sonnet-20241022",
                "model": "claude-3-7-sonnet-20250219",
                "api_key": os.getenv("ANTHROPIC_API_KEY"),
                "api_type": "anthropic",
            }
        ]

        llm_config = {"config_list": config_list_claude, "cache_seed": cache_seed}

        self.user_proxy = autogen.UserProxyAgent(
            name="User_proxy",
            system_message="A human admin.",
            code_execution_config={
                "last_n_messages": 2,
                "work_dir": "groupchat",
                "use_docker": False,
            },  
            human_input_mode="NEVER",
            is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
        )

        self.coordinator = autogen.AssistantAgent(
            name="coordinator",
            system_message=
            f'''
            You are a coordinator who can help the user find the correct corresponding ASCO guidelines.
            You have access to a dictionary whose keys are the guideline numbers and values are descriptions of the guidelines: {guidelines_to_use}
            Based on the user's prompt, you will determine which ASCO guideline to use, 
            and then return the key as a string (e.g. breast_cancer_8) and the user prompt, and ask pdf_viewer to retrieve information from the pdf.
            If none of the ASCO guidelines are relevant, return "none" and ask User_proxy to terminate.

            - If a relevant guideline is found:
                suggested tool call augments would be:
                "key": [insert guideline key here]
                "prompt": [insert original user query here and add "must also include the exact context of each point."]

            - If no relevant guideline is found:
                <result>
                guideline_key: none
                explanation: [insert your explanation here]
                action: User_proxy, please terminate the process.
                </result>
            ''',
            llm_config=llm_config,
        )
    
        self.pdf_viewer = autogen.AssistantAgent(
            name="pdf_viewer",
            system_message="You are designed to answer questions based on the content of a specific PDF document. Also retrieve the references from the pdf and include them in the answer.",
            llm_config=llm_config,
        )

        self.reviewer = autogen.ConversableAgent(
            name="reviewer",
            system_message=
            '''
            You are a reviewer who review the answer from pdf_viewer and provide a final answer to the user's question.
            Instructions:
            1. Review the user's question. 
            2. Carefully read the pdf_viewer output, including the answer and the context of the answer.
            3. Identify key information relevant to the user's question.
            4. If the user's question is not answered, ask pdf_viewer to go back and find the correct answer.
            5. If the user's question is answered, formulate a clear and concise final answer to the user's question, include the references in the answer, and add TERMINATE at the end of the answer.
            ''',
            llm_config=llm_config,
        )

        # Register the Claude PDF processing tool
        register_function(
            process_pdf, 
            caller=self.coordinator,
            executor=self.pdf_viewer,
            name="process_pdf", 
            description="Retrieve information from pdf.")

        self.allowed_transitions = {
            self.user_proxy: [self.coordinator],
            self.coordinator: [self.pdf_viewer],
            self.pdf_viewer: [self.reviewer],
        }

        self.groupchat = autogen.GroupChat(
            agents=[self.user_proxy, self.coordinator, self.pdf_viewer, self.reviewer], 
            allowed_or_disallowed_speaker_transitions=self.allowed_transitions,
            speaker_transitions_type="allowed",
            messages=[],
            max_round=6,
        )
        self.manager = autogen.GroupChatManager(groupchat=self.groupchat, llm_config=llm_config)

    def chat(self, message):
        return self.user_proxy.initiate_chat(self.manager, message=message)

# Usage:
# claude_chat = ClaudeChat()
# chat_res = claude_chat.chat("How to give adjuvant pembro with radiation therapy for patients with localized triple-negative breast cancer?")