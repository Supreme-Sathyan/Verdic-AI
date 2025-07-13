# To run this code you need to install the following dependencies:
# pip install google-genai
import base64
import os
from google import genai
from google.genai import types
from extract import extract_pdf_as_one_paragraph
from docgen import generate_legal_brief_pdf_from_json

def generatedoc():
    client = genai.Client(
        api_key=""
    )

    model = "gemini-2.5-pro-exp-03-25"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="""i will give a pargraph for with document detaisl
u analyze it and classify it to following categories for a legal brief documnet 
"Legal Brief": {
    "case_name": "",
    "case_number": "",
    "plaintiff": "",
    "defendant": "",
    "court_name": "",
    "date_filed": "",
    "date_of_order": "",
    "summary_of_facts": "",
    "legal_arguments": "",
    "conclusion": ""

ex - In case number 12345, the plaintiff, John Doe, has filed a lawsuit against the defendant, Jane Smith, alleging that she breached a contract they had entered into regarding the sale of goods. The plaintiff claims that the defendant failed to deliver the goods as agreed, causing financial loss and inconvenience. As a result, John Doe seeks compensation for the damages incurred due to the defendant’s failure to fulfill the contract. In defense, Jane Smith argues that the contract was invalid due to a misunderstanding of the terms, and she denies any breach of agreement. She contends that she should not be held liable for the alleged damages and seeks the dismissal of the case.

do it for above para
give as json file format
"""),
            ],
        ),
        types.Content(
            role="model",
            parts=[
                types.Part.from_text(text="""Here's my thinking process to arrive at the JSON output:

1.  *Understand the Goal:* The user wants me to extract specific pieces of information (Case Number, Plaintiff, Defendant, Summary, Claim, Defense) from a provided paragraph describing a legal case and format this information as a JSON object.

2.  *Analyze the Request Structure:*
    *   Input: A paragraph detailing a legal case.
    *   Task: Analyze the paragraph and identify the key elements corresponding to the predefined categories.
    *   Output Format: A JSON object where the keys are the predefined categories (Case Number, Plaintiff, Defendant, Summary, Claim, Defense) and the values are the extracted text corresponding to each category.
    *   Example Provided: The user gave an example paragraph and asked me to process that specific example paragraph. This is important – I need to apply the rules to the provided text, not just understand the concept.

3.  *Break Down the Categories and Identify Keywords/Phrases:*
    *   *Case Number:* Look for phrases like \"case number\", \"docket number\", or specific numerical identifiers presented as such.
    *   *Plaintiff:* Look for terms like \"plaintiff\", \"claimant\", \"filed a lawsuit against\", or the party initiating the action.
    *   *Defendant:* Look for terms like \"defendant\", \"respondent\", \"sued by\", \"against\", or the party being sued.
    *   *Summary:* This requires synthesizing the core conflict. What is the lawsuit about in a nutshell? It often involves both the plaintiff's allegation and the general context.
    *   *Claim:* What does the plaintiff specifically allege the defendant did wrong? What outcome does the plaintiff seek? Look for phrases like \"alleges that\", \"claims that\", \"seeks compensation for\", \"due to the defendant's failure\".
    *   *Defense:* What is the defendant's response to the claim? What arguments do they make? Look for phrases like \"in defense\", \"argues that\", \"denies\", \"contends that\", \"seeks the dismissal\".

4.  *Process the Example Paragraph Step-by-Step:*

    *   *Paragraph:* \"In case number 12345, the plaintiff, John Doe, has filed a lawsuit against the defendant, Jane Smith, alleging that she breached a contract they had entered into regarding the sale of goods. The plaintiff claims that the defendant failed to deliver the goods as agreed, causing financial loss and inconvenience. As a result, John Doe seeks compensation for the damages incurred due to the defendant’s failure to fulfill the contract. In defense, Jane Smith argues that the contract was invalid due to a misunderstanding of the terms, and she denies any breach of agreement. She contends that she should not be held liable for the alleged damages and seeks the dismissal of the case.\"

    *   *Case Number:* The text explicitly states \"In case number 12345\". So, Case Number = 12345.
    *   *Plaintiff:* The text explicitly states \"the plaintiff, John Doe\". So, Plaintiff = John Doe.
    *   *Defendant:* The text explicitly states \"against the defendant, Jane Smith\". So, Defendant = Jane Smith.
    *   *Summary:* What's the overall picture? John Doe sued Jane Smith over a breached contract for the sale of goods. So, Summary = John Doe sued Jane Smith for an alleged breach of contract regarding the sale of goods.. (Slight rephrasing for conciseness is okay for a summary).
    *   *Claim:* What does John Doe allege and want? He \"alleges that she breached a contract\", \"claims that the defendant failed to deliver the goods as agreed, causing financial loss and inconvenience\", and \"seeks compensation for the damages incurred\". So, Claim = The defendant breached a contract by failing to deliver goods as agreed, causing financial loss. The plaintiff seeks compensation for damages..
    *   *Defense:* What is Jane Smith's response? \"In defense, Jane Smith argues that the contract was invalid due to a misunderstanding of the terms, and she denies any breach of agreement. She contends that she should not be held liable for the alleged damages and seeks the dismissal of the case.\" So, Defense = The contract was invalid due to a misunderstanding of terms; there was no breach of agreement. The defendant denies liability and seeks dismissal of the case..

5.  *Format as JSON:* Now, assemble the extracted information into the requested JSON format.

    
    {
  "Legal Brief": {
    \"case_name\": \"/2019\",
    \"case_number\": \"/2019\",
    \"plaintiff\": \"Smt.Padm\",
    \"defendant\": \"M/s Sree Krishna Automotives Hyd Pvt. Ltd.\",
    \"court_name\": \"DISTRICT CONSUMER DISPUTES REDRESSAL COMMISSION: / RANGA REDDY\",
    \"date_filed\": \"05.09.2019\",
    \"date_of_order\": \"14.02.2023\",
    \"summary_of_facts\": \"Complainant purchased a Honda Amaze / from the Opposite Party. / Complainant alleged the vehicle had defects / and was previously sold to another customer. / Complainant seeks remedies for unfair trade practice / and deficiency of service.\",
    \"legal_arguments\": \"Complainant argued unfair trade practice / due to the sale of a previously sold vehicle / and defects in the vehicle. / Opposite Party denied the allegations, / stating the previous customer's details / were inadvertently uploaded / and the defects were not significant.\",
    \"conclusion\": \"Complaint partly allowed. / Opposite Party directed to rectify defects, / extend warranty, / and pay compensation / for physical hardship, / financial and mental trauma.\"
  }
}
    

6.  *Final Review:* Does the JSON accurately reflect the information in the paragraph according to the requested categories? Yes. Is it correctly formatted as JSON? Yes. The process is complete."""),
                types.Part.from_text(text="""
{
  "Legal Brief": {
    \"case_name\": \"/2019\",
    \"case_number\": \"/2019\",
    \"plaintiff\": \"Smt.Padm\",
    \"defendant\": \"M/s Sree Krishna Automotives Hyd Pvt. Ltd.\",
    \"court_name\": \"DISTRICT CONSUMER DISPUTES REDRESSAL COMMISSION: / RANGA REDDY\",
    \"date_filed\": \"05.09.2019\",
    \"date_of_order\": \"14.02.2023\",
    \"summary_of_facts\": \"Complainant purchased a Honda Amaze / from the Opposite Party. / Complainant alleged the vehicle had defects / and was previously sold to another customer. / Complainant seeks remedies for unfair trade practice / and deficiency of service.\",
    \"legal_arguments\": \"Complainant argued unfair trade practice / due to the sale of a previously sold vehicle / and defects in the vehicle. / Opposite Party denied the allegations, / stating the previous customer's details / were inadvertently uploaded / and the defects were not significant.\",
    \"conclusion\": \"Complaint partly allowed. / Opposite Party directed to rectify defects, / extend warranty, / and pay compensation / for physical hardship, / financial and mental trauma.\"
  }
}"""),
            ],
        ),
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="""INSERT_INPUT_HERE"""),
            ])
        
    ]
    run = True
    while run:
        
        user_input = extract_pdf_as_one_paragraph("j.pdf")+"give legal brief json"

        # Append user message
        contents.append(types.Part(text=user_input))
 
        # Configure generation parameters
        generate_content_config = types.GenerateContentConfig(
            
            response_mime_type="text/plain",
        )

        # Get model response
        response_text = ""
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            response_text += chunk.text

        
        filepath=generate_legal_brief_pdf_from_json(response_text)
        print("legal brief stored in: "+filepath)
        
        

        # Print and add model response to history
        
        contents.append(types.Part(text=user_input))
        run = False

if __name__ == "__main__":
    generatedoc()
