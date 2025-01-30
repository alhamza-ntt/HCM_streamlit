from openai import AzureOpenAI
import openai
from config import GPT_CONFIG



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