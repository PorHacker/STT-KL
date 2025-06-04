"""
Auto qa list v2 적용
- 평가항목 상세내용에 있는 예시 멘트 중 하나를 발화했는지 확인. 
- 평가결과는 긍정 또는 부정 이진으로 나뉨. 
- 프롬프트는 instruction, 평가항목상세, input call text, output format 으로 이루어짐. 
"""

# 모든 평가항목 프롬프트에 공통으로 들어가는 부분.
INSTRUCTION = """You are an evaluator to rate agents in a customer center, from e-commerce shop called 코오롱몰. You are given a transcript between a customer and an agent and a criterion to assess the agent. 코오롱몰 is e-commerce website that sales various fashion products from multiple brands. In the transcript, agent handles issue that customers encounter while using 코오롱몰. Your job is to evaluate agent's utterances or action based on the given criterion. 
Read the criterion and transcript carefully to conduct accurate assessment. Note that some parts in the dialogue may be replaced with '*', as it's customer's private information. Also, inaccurate text recognition may appear in the transcript as it's transcribed from speech record. Do not consider this as agent's fault while evaluation. """

# 전체 프롬프트 포맷
PROMPT_TEMPLATE = """# INSTRUCTION
{instruct}

Your job is to generate following 3 contents for evaluation :
1. remark
- Find agent's remark directly from given 'TRANSCRIPT' that is same with one of following Example sentences from CRITERION. 
- If multiple remarks appear, write as list. If no remarks exist to extract, write None. 
- Make sure each remark is directly extracted from given TRANSCRIPT.
2. assessment
{assessment_type}
3. comment
- Write concise rationale why you resulted in certain assessment, in Korean.

# CRITERION
{criterion}
Example sentences:
{examples}

# TRANSCRIPT
{input}

# OUTPUT
Include only following json object in your answer. Make sure to generate accurate json format.
{{'assessment':(string), 'remark':(list), 'comment':(string)}}
"""


# 고객 컴플레인 prompt template
PROMPT_TEMPLATE_COMPLAIN = """# INSTRUCTION
{instruct}

Your job is to generate following 3 contents for evaluation :
1. remark
- A remark is a list of sentences that satisfy all of the following conditions. If no remarks exist to extract, write None. 
- The remark sentence must be actually uttered by customer, from TRANSCRIPT.
- The remark sentence must exactly match one of the Example sentences given in the CRITERION. 
- You shouldn't include a sentence as remark simply because it's a customer complaining, but only if it exactly matches the example sentence given. 
2. assessment
{assessment_type}
3. comment
- Write concise rationale why you resulted in certain assessment, in Korean.

# CRITERION
{criterion}
Example sentences:
{examples}

# TRANSCRIPT
{input}

# OUTPUT
Include only following json object in your answer. Make sure to generate accurate json format.
{{'assessment':(string), 'remark':(list), 'comment':(string)}}
"""


# OUTPUT_POSITIVE = """- There exists agent's remark from TRANSCRIPT which is same with one of the example sentences provided in the CRITERION, put 'Positive'. Otherwise, put 'Negative'. """

# OUTPUT_NEGATIVE = """- There exists agent's remark from TRANSCRIPT which is same with one of the example sentences provided in the CRITERION, put 'Negative'. Otherwise, put 'Positive'."""

# OUTPUT_POSITIVE = """- There exists agent's remark from TRANSCRIPT which is similar to one of the example sentences provided in the CRITERION, put 'Positive'. Otherwise, put 'Negative'. """

# OUTPUT_NEGATIVE = """- There exists agent's remark from TRANSCRIPT which is simlilar to one of the example sentences provided in the CRITERION, put 'Negative'. Otherwise, put 'Positive'."""

OUTPUT_POSITIVE = """- There exists agent's remark from TRANSCRIPT which is identical to one of the example sentences provided in the CRITERION, put 'Positive'. Otherwise, put 'Negative'. """

OUTPUT_NEGATIVE = """- There exists agent's remark from TRANSCRIPT which is identical to one of the example sentences provided in the CRITERION, put 'Negative'. Otherwise, put 'Positive'."""


# 필수안내사항에 해당하는 항목 분류
REQUIREMENTS_CLS = """# INSTRUCTION
You are a given transcript between a customer and an agent at a customer center from 코오롱몰. 코오롱몰 is e-commerce website that treats various fashion brands and products. There are multiple inquirys from customer, like shipping, ordering, refund and more. Note that the transcript is transcribed from call record, so inaccurate text may appear. 

# You must select a number from the list of inquiry types. Follow these guidelines:
- You are given a list of inquiry types and the conditions that corresspond to each inquiry type. 
- The condition must met at the time of the call between the customer and agent. 
- Select one inquiry type from the list only if the condition is fully satisfied by the transcript. Don't choose the inquiry type simply because it's relevant. 
- If none of the list corresponds with the transcript, output '0'. 
- Review the selected inquiry type and it's condition again thoroughly, and check if the condition is completely satisfied by the given transcript, otherwise, please output '0'.

# LIST OF INQUIRY TYPES
{data}

# TRANSCRIPT
{input}

# OUTPUT
Include only single number from the list."""


# 필수안내사항
REQUIREMENTS = """# INSTRUCTION
{instruct}

Depending on the customer's inquiry, there are a certain keywords that must be mentioned from the agent. This content is called 필수안내, and this criterion evaluates how much of this 필수안내 is well delivered to the customer. 
Your job is to generate following 3 contents for evaluation :
1. remark
- Extract any utterance of agent from TRANSCRIPT, which contains one of the keywords given below in CRITERION. 
- If multiple remarks appear, write as list. If no remarks exist to extract, write 'None'. 
2. assessment
- If every keywords from the criterion are mentioned from agent, put 'Positive'.
- If one or more keywords are missing, put 'Negative'.
3. comment
- Write concise rationale why you resulted in certain assessment, in Korean.

# CRITERION
Depending on the customer's inquiry, there are certain contents that must be mentioned from the agent. This criterion evaluates how much of this contents are well delivered to the customer. 
Keywords :
{keywords}
Examples :
{examples}

# TRANSCRIPT
{input}

# OUTPUT
Include only following json object in your answer. Make sure to generate accurate json format. 
{{'remark':(list), 'assessment':(string), 'comment':(string)}}
"""
