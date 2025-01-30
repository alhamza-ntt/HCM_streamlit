from openai import AzureOpenAI
import openai
from config import ADA_CONFIG, GPT_CONFIG
from config import app_logger
import requests
import ast
import re
import nltk

import requests
import uuid
import json
from typing import Optional, Dict, Union

#nltk.download('punkt')  
from nltk.tokenize import sent_tokenize


def get_embedding(input_string):
   
    openai_client = AzureOpenAI(
        api_key=ADA_CONFIG["api_key"],
        api_version=ADA_CONFIG["api_version"],
        azure_endpoint=ADA_CONFIG["api_base"],
        azure_deployment=ADA_CONFIG["deployment_name"],
    )
    response = openai_client.embeddings.create(
        input=input_string,
        model=ADA_CONFIG["model"],
    )
    results = response.data[0].embedding
    openai_client.close()
    return results


def vectroize(dataList):
    

    data_list = []

    for i in range(len(dataList)):  
        insatnce = dataList[i]
        vector = get_embedding(insatnce)
        app_logger.info(f"vector number {i+1}/{len(dataList)} was added sucessfuly!")
        data_dict = {"id" : str(i+44) , 'content': insatnce, 'contentVector': vector}
        data_list.append(data_dict)
    return  data_list




def get_completion(prompt, temperature=0.7, top_p=0.95, frequency_penalty=0, presence_penalty=0,
                   verbose_token=False):
  
    openai_client = AzureOpenAI(
        api_key=GPT_CONFIG["api_key"],
        api_version=GPT_CONFIG["api_version"],
        azure_endpoint=GPT_CONFIG["api_base"],
        azure_deployment=GPT_CONFIG["deployment_name"],
    )
    try:
        response = openai_client.chat.completions.create(
            model=GPT_CONFIG["model"],
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )
        if verbose_token:
            app_logger.info(f"(OpenAI/GPT Token Usage): Prompt: {response.usage.prompt_tokens} + Completion: "
                            f"{response.usage.completion_tokens} = Total: {response.usage.total_tokens}")
        return response.choices[0].message.content
    except openai.APIConnectionError as e:
        openai_client.close()
        app_logger.info(f"(OpenAI/GPT): Failed to connect to OpenAI API: {e}")
        return None
    except openai.APIError as e:
        openai_client.close()
        app_logger.info(f"(OpenAI/GPT): OpenAI API returned an API Error: {e}")
        return None
    except openai.RateLimitError as e:
        openai_client.close()
        app_logger.info(f"(OpenAI/GPT): OpenAI API request exceeded rate limit: {e}")
        return None
    



def prepare_promt(question , context):

   finalprompt =f"""you are a question answering system, given a question and related
    pages/chunks, your task is to give the final result.
     
    here is the question:
        {question}.
        
        
    and here are the realted pages:
        {context}
        
        """
   return finalprompt





def parse_data(content):
    prompt = (
        f"Parse the given result of web scraping a restaurant into line-by-line data. "
        "I don't need all the information, just the content of the menu with their categories and the prices: "
        f"{content}"
    )
    return get_completion(prompt=prompt)

def ConvertToList(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        return ast.literal_eval(content)
    




#read the txt file and convert it to a list split on produktname
def read_and_list(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        # Read the whole content of the file
        text = file.read()
    products = re.split(r'(Produktname:)', text)[1:]  # We discard the first empty element, if present

    # Combine 'Produktname' back to each product
    products = [''.join(products[i:i+2]) for i in range(0, len(products), 2)]


    return products




def split_product_description(description, produktname, word_limit=400):
    """
    split the data into chunks of size word_limit and add the prdocuktname at the start of each chunk
    
    
    
    """
    # Tokenize description into sentences
    sentences = sent_tokenize(description)
    
    # Calculate number of chunks needed
    total_words = len(description.split())
    num_chunks = (total_words // word_limit) + (1 if total_words % word_limit != 0 else 0)

    # Calculate chunk size (aiming for equal distribution)
    target_chunk_size = total_words // num_chunks

    chunks = []
    current_chunk = [produktname]  # Each chunk starts with Produktname
    current_chunk_word_count = len(produktname.split())

    for sentence in sentences:
        sentence_word_count = len(sentence.split())
        
        # Check if adding this sentence would exceed the target chunk size
        if current_chunk_word_count + sentence_word_count > target_chunk_size and len(chunks) < num_chunks - 1:
            # Finalize current chunk and start a new one
            chunks.append(' '.join(current_chunk))
            current_chunk = [f"{produktname} : \n"]  # Start new chunk with Produktname
            current_chunk_word_count = len(produktname.split())

        # Add sentence to the current chunk
        current_chunk.append(sentence)
        current_chunk_word_count += sentence_word_count

    # Append the final chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks




def vectroizeForPincone(dataList):
    
    data_list = []

    for i in range(len(dataList)):  
        insatnce = dataList[i]
        vector = get_embedding(insatnce)
        #app_logger.info(f"vector number {i+1}/{len(dataList)} was added sucessfuly!")
        data_dict = {"id" : str(i) , 'values': vector, 'metadata': dict({"chunk" : insatnce})}
        data_list.append(data_dict)
    return  data_list





def translate_text(text: str, target_lang: str = 'de', source_lang: str = 'de') -> Optional[Dict[str, str]]:
    """
    Translates text to the specified target language using Microsoft's Translator API.
    
    Args:
        text (str): The text to translate
        target_lang (str): The language code to translate to (e.g., 'en', 'es', 'fr')
        source_lang (str): The source language code (defaults to 'de')
    
    Returns:
        Optional[Dict[str, str]]: Dictionary containing status and translation/error message
    """
    if not text or not target_lang:
        return {
            "status": "ERROR",
            "error": "Text and target language are required!"
        }
    
    try:
        # Azure Translator API configuration
        key = "EZ5fOJVKxKW2tCBJQhzSwg5kkCJLs5R9ii1kjetkGtTENLZm4xV1JQQJ99AKACfhMk5XJ3w3AAAbACOGbhNk"
        endpoint = "https://api.cognitive.microsofttranslator.com"
        location = "swedencentral"
        path = '/translate'
        
        # Construct URL and parameters
        constructed_url = endpoint + path
        params = {
            'api-version': '3.0',
            'from': source_lang,
            'to': [target_lang]
        }
        
        # Set up headers
        headers = {
            'Ocp-Apim-Subscription-Key': key,
            'Ocp-Apim-Subscription-Region': location,
            'Content-type': 'application/json',
            'X-ClientTraceId': str(uuid.uuid4())
        }
        
        # Prepare request body
        body = [{
            'text': text
        }]
        
        # Make translation request
        response = requests.post(constructed_url, params=params, headers=headers, json=body)
        response.raise_for_status()  # Raise exception for bad status codes
        
        # Parse response
        translation_result = response.json()
        translated_text = translation_result[0]["translations"][0]["text"]
        app_logger.info("the translation API is triggered")
        return translated_text
        
        
    except requests.exceptions.RequestException as e:
        return {
            "status": "ERROR",
            "error": f"Translation API request failed: {str(e)}"
        }
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        return {
            "status": "ERROR",
            "error": f"Failed to parse translation response: {str(e)}"
        }
    


def system_prompt_HCM(user):
    context = f"""


**Objective:**  
Generate a comprehensive and structured salary report for an employee, including a proposed salary, salary band, justification for the recommendations, and a detailed salary development plan for the next three years. The report should be based on the employee's current salary, role, experience, market comparisons, and performance metrics. Do not include actual numbers in the output; instead, use placeholders (e.g., [Current Salary], [Proposed Salary], [Market Average]) to indicate where data should be inserted.

---

**Input Data:**  
1. **Employee Details:**  
   - Current Role: [Role]  
   - Current Salary: [Current Salary]  
   - Years of Experience: [Years of Experience]  
   - Tenure at Company: [Tenure]  
   - Performance Metrics: [Performance Metrics]  

2. **Market Comparison Data:**  
   - Average Market Salary: [Market Average]  
   - Average Salary by Education Level: [Education Average]  
   - Average Salary by Experience: [Experience Average]  
   - Average Salary by Job Profile: [Job Profile Average]  
   - Average Salary by Location: [Location Average]  
   - Cost of Living Adjustment: [Cost of Living Adjustment]  

3. **Salary Trends:**  
   - Annual Salary Growth: [Annual Growth Rate]  

---
here is the expected output format:
**Report Structure:**  

1. **Salary Recommendation:**  
   - Proposed Salary: [Proposed Salary based on given data]  
   - Proposed Salary Band: [Salary Band Range based on given data]  

2. **Justification for Recommendations:**  
   - Compare the employee's current salary with market averages (education, experience, job profile, location).  
   - Highlight the employee's performance, role, and tenure as justifying factors for the proposed salary.  
   - Address any discrepancies, such as the current salary being below the cost of living or above certain market averages.  

3. **Salary Development Plan:**  
   - Provide a year-by-year salary projection for the next three years, including percentage increases.  
   - Justify the increases based on expected performance, market trends, and potential career advancements (e.g., promotions, additional certifications).  

4. **Chain of Thought:**  
   - Break down the reasoning behind each recommendation step-by-step.  
   - Include analysis of market data, employee-specific factors, and long-term considerations.  


---  
fill in the placeholders with the relevant data and keep the format consistent.
do not retuen report with placeholders, only the final report with the data filled in.

and here are the user details:
{user}

"""
    return context 