INSTRUCTION = """You are given a transcript between customer and agent at a customer service center of 코오롱몰. 코오롱몰 is e-commerce fashion mall that treats multiple fashion brands. The service center deals with customer's order, inquiry about a product, delivery, refund, exchange and all inquiries from customer's experience at 코오롱몰. 
You are to conduct following task based on the dialogue. Note that some text from dialogue are masked with placeholder '*', due to customer's privacy issue and you must ignore these parts. """

CHAT_INSTRUCTION = """You are given a dialogue between customer and agent's chat history at a customer service center of 코오롱몰. 코오롱몰 is e-commerce fashion mall that treats multiple fashion brands. The service center deals with customer's order, inquiry about a product, delivery, refund, exchange and all inquiries from customer's experience at 코오롱몰. 
You are to conduct following task based on the dialogue. Note that some text from dialogue are masked with placeholder '*', due to customer's privacy issue and you must ignore these parts. """


### Realtime tasks


# category classification
CATEGORY_CLS = """# INSTRUCTION
{instruct}
Consider 상품 and 제품 as same word. 

# Your job is to choose which topic from the list of category is the most related with the transcript. 

# TRANSCRIPT
{input}

# LIST OF CATEGORY
{data}

# OUTPUT
Include only corresponding single topic number from topic list in your output. """


# 브랜드 키워드 추출
BRAND_EXTRACT = """# INSTRUCTION
{instruct}

# Your job is to extract brand name which is directly mentioned in given transcript.
Follow this guideline for your job:
1. Review given list of brand, and find the brand name that is directly mentioned in the given dialogue. Output corresponding number from the list. 
2. If none in the list of brand can be found in the dialogue, output 'None'.
3. Note that 코오롱몰 is NOT a brand name, it is online shopping website name. Do not confuse 코오롱몰 and 코오롱스포츠 as they are different names. Output 코오롱스포츠 as brand name only if 코오롱스포츠 is explicitly mentioned. 

# DIALOGUE
{input}

# LIST OF BRAND
{data}

# OUTPUT
Include only single number from the list in your output. """


# 상담사, 고객 발화 구분 요약
SPEAKER_SPECIFIC_SUMMARY = """# INSTRUCTION
{instruct}

# Your job is to generate following two summaries.
1. Customer summary : Based on customer's remark, identify customer's inquiry, needs, requests or subject of issues in detail.
2. Agent summary : Based on agent's remark, identify the action, guidance or response the agent took on the customer's issue.

Guidelines for summary:
- Exclude typical opening or closing remarks as greeting or gratitude. Also leave out trivial details that are off the main conversation.
- Write a descriptive, specific, coherent summary in concise words.
- Write in a brief, report-like style, using short and simple sentences as shown in EXAMPLE. Use "~함, ~했음" instead of "~하였다, ~했다, ~했습니다" in the end of the sentence.
- Make sure that summary is written in Korean.

# EXAMPLE
{{'customer': '로고 포인트 올베스트를 엑스 투엑스라지에서 엑스라지로 교환 신청했으나, 교환 반품 완료 문자를 받았음에도 아직 상품을 받지 못했다고 이의를 제기함. ',
'agent': '로고 포인트 올베스트는 엑스라지로 교환 접수되었고, 이미 입고 처리가 완료되었으므로 곧 출고 작업이 진행될 것이라고 안내함.'}}

# INPUT DIALOGUE
{input}

# Output json object with following information :
{{"customer":(customer summary in concise text), "agent":(agent summary in concise text)}}"""


### Batch tasks

# 주제 생성
TITLE_GENERATION_0 = """# INSTRUCTION
{instruct}

Your job is to generate a title of the transcript by following below guideline. 
- Include customer's main inquiry or request, and include reason for the request.
- Include brand and product name if mentioned, which is directly related to the main request and mainly discussed.
- Keep the title as descriptive and concise as possible while including all essential information.
- Title must be Korean. 

# Example
- If a specific reason for the request is clear: "[Brand Name] [Product Name] [Main Customer Request(Request Reason)]" e.g. "{{'title':'드라이트 위캔드 더블백 배송 확인 요청(배송 지연)'}}", "{{'title':'골프화 교환 요청(사이즈 불일치)'}}"
- If no specific reason is given: "[Brand Name] [Product Name] [Main Customer Request]" e.g. "{{'title':'서핑다트 반팔 티셔츠 재고 확인 요청'}}", "{{'title':'센소나이트 가방 주문 상태 확인'}}"
- If the brand name or product name is not explicitly mentioned for the item in question, use only the information available.

# TRANSCRIPT
{input}

# OUTPUT
Include only following json object in your output. Please make accurate json format. 
{{'title':'(string)'}}"""


TITLE_GENERATION = """# INSTRUCTION
{instruct}

Your job is to generate a title of the transcript by following below guideline. 
- Include customer's main inquiry or request, and include reason for the request.
- Include brand, product name and product code if mentioned, which is directly related to the main request and mainly discussed. A product code consists of English letters and numbers. 
- Keep the title as descriptive and concise as possible while including all essential information.
- Title must be Korean. 

# Example
- If a specific reason for the request is clear: "[Brand Name] [Product Name and Product Code] [Main Customer Request(Request Reason)]" e.g. "드라이트 위캔드 더블백 배송 확인 요청(배송 지연)", "골프화 교환 요청(사이즈 불일치)"
- If no specific reason is given: "[Brand Name] [Product Name and Product Code] [Main Customer Request]" e.g. "서핑다트 반팔 티셔츠 재고 확인 요청", "센소나이트 가방 주문 상태 확인"
- If the brand name, product name, product code is not explicitly mentioned for the item in question, fill out above format using only information available. 

# TRANSCRIPT
{input}

# OUTPUT
Include only following json object in your output. Please make accurate json format. 
{{'title':'(string)'}}"""


# 상담결과 고객감정 긍정/부정/중립 분류
CUSTOMER_SENTIMENT_CLS = """# INSTRUCTION
{instruct}

# Your job is to classify customer's sentiment after a conversation with an agent. Choose customer's sentiment among "Positive", "Neutral", or "Negative". 

A "Positive" sentiment means the customer's issue or concern was resolved satisfactorily by the agent, and the conversation progressed positively overall.
A "Neutral" sentiment means the customer's issue or concern was partially resolved, or the conversation did not clearly convey a positive or negative sentiment.
A "Negative" sentiment means the customer's issue or concern was not resolved, or the customer displayed negative behavior such as frustration, anger, or requesting a different agent, regardless of any polite closing remarks or gratitude.

To accurately classify the customer's sentiment, consider the following guidelines:
1. Read the entire dialogue carefully, paying close attention to the customer's language, tone, and expressions throughout the conversation.
2. Focus on the overall progression of the conversation and whether the customer's issue was effectively resolved.
3. Do not solely rely on any polite closing remarks made by the customer at the end.

# DIALOGUE
{input}

# Your Answer:
('Positive' or 'Neutral' or 'Negative')"""


# 상품 키워드 추출
KEYWORD_EXTRACT = """# INSTRUCTION
{instruct}

# Your job is to extract 3 different keywords from the transcript. 
Follow this guideline for your job:
1. Read through the transcript and indentify customer's inquiry, request, or demand. 
2. You need to select three different keywords that reflect the customer's issues. The keywords must be in Korean and mentioned directly in the conversation.
3. Don't include personal information such as your name or contact information as keywords.
4. Focus on words or phrases that represent the key issues, products, or services discussed in the conversation.
5. If multiple topics were discussed, prioritize the most important or recurring topics.
6. Review the keywords you've selected to ensure that they don't duplicate meaning and accurately summarize the main points of the conversation. 

# DIALOGUE #

{input}

# OUTPUT
Include only following json object in your output. 
{{
    "keyword1":(str),
    "keyword2":(str),
    "keyword3":(str)
}}
"""

KEYWORD_EXTRACT = """# INSTRUCTION
{instruct}

# Your job is to extract 3 different keywords from the transcript. 
Follow this guideline for your job:
1. Read through the transcript and indentify customer's inquiry, request, or demand. 
2. You need to select three different keywords that reflect the customer's issues. The keywords must be in Korean and mentioned directly in the conversation.
3. Don't include personal information such as your name or contact information as keywords.
4. Focus on words or phrases that represent the key issues, products, or services discussed in the conversation.
5. If multiple topics were discussed, prioritize the most important or recurring topics.
6. Review the keywords you've selected to ensure that they don't duplicate meaning and accurately summarize the main points of the conversation. 

# DIALOGUE #
{input}

# OUTPUT
Include only following json object in your output. 
{{
    "keyword1":(str),
    "keyword2":(str),
    "keyword3":(str)
}}
"""
