"""
Preprocess chat, couns(board) channel data and apply masking
v2 update 20240715
"""

import re
import json
import logging

# Create a logger instance
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
# stream_handler = logging.StreamHandler()
# logger.addHandler(stream_handler)

count_remove_templates = [
    "고객센터 운영시간에는 아래 안내드린 상담채널을 이용하시면 실시간 상담 가능합니다.",
    "\n◈ 전화 상담 ◈☎ 1588-7667 (상담사 운영시간 : 09:30 ~ 18:00 / 공휴일, 주말 제외)(유료)",
    "\n◈ 채팅 상담 ◈코오롱몰 우측 하단에 채팅상담 버튼 클릭 (상담사 운영시간 : 09:30 ~ 18:00 / 공휴일, 주말 제외)",
    "만족도조사 바로가기",
]


def preprocess_call(text):
    # Define regular expression pattern
    dialogue_pattern = r"{(고객|상담사):(.*?)}"

    # Extract matching parts based on pattern
    dialogues = re.findall(dialogue_pattern, text)

    extracted_texts = []

    for role, content in dialogues:
        if content:
            # Extract Korean text and add to the list
            korean_text = re.sub("[^가-힣\s:.,*!?0-9]", "", role + ":" + content)
            if korean_text != "":
                extracted_texts.append(korean_text.strip())

    return extracted_texts


def preprocess_chat(org):
    """채팅 데이터 전처리"""
    text = re.sub(r"{상담사:T}\r,", "", org)

    # 2024.08.06: Update for all empy chat content
    text = re.sub(r"{상담사:T}\r", "", text)

    conv = text.split("\r,")
    conv = [re.sub(r"\r+", "", x) for x in conv]
    conv = [x[1:-1] for x in conv]

    conv_merge = []
    speaker = "상담사"
    remark = ""
    for x in conv:
        idx = x.find(":")
        s = x[:idx]
        r = x[idx + 1 :]
        if s == speaker:
            remark += r + " "
        else:
            if len(remark) > 0:
                conv_merge.append(f"{speaker}:{remark}")
            speaker = s
            remark = r
    conv_merge.append(f"{speaker}:{remark}")

    result = "\n".join(conv_merge)
    result = re.sub(r"\n+", "\n", result)
    result = re.sub(r"https://\S+", "(url)", result)  # url link convert
    result = re.sub(r"[^\S\r\n]+", " ", result)
    # breakpoint()
    return result


# def preprocess_couns(org):
#     """게시판 데이터 전처리"""

#     text = re.sub("&nbsp;", " ", org)
#     idx = text.find("[답변]")
#     customer = text[4:idx]
#     customer = re.sub(r"\r", "", customer)  # remove carriage return
#     agent = text[idx + 4 :]
#     agent = re.sub(r"<.*?>", "", agent)  # remove html tags
#     agent = re.sub(r"\r", "", agent)
#     agent = re.sub("만족도조사 바로가기", "", agent)

#     result = f"고객:{customer}\n상담사:{agent}"
#     result = re.sub(r"[^\S\r\n]+", " ", result)
#     return result


def remove_templates(text, templates=count_remove_templates):
    # Loop through each template and remove it from the text
    for template in templates:
        text = text.replace(template, "")  # Replace the template with an empty string
    return text


def preprocess_couns(org):
    """게시판 데이터 전처리"""

    text = re.sub("&nbsp;", " ", org)

    # replace <br> tags with newline characters
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)  # (?i) makes it case-insensitive

    # Then remove all other HTML tags
    clean = re.compile("<.*?>")
    result = re.sub(clean, "", text)

    # remove unnecessary text templates
    result = remove_templates(result, templates=count_remove_templates)

    # remove multiple newlines
    result = re.sub(r"\n+", "\n", result)
    result = re.sub(r"\\n+", "\n", result)

    return result


def preprocess(text, in_gb):
    """
    Main preprocessing function that routes the input text to the appropriate preprocessing function.

    Args:
        text (str): Raw text data.
        in_gb (str): Indicator of the data type. Must be one of ["CALL", "CHAT", "BOARD"].

    Returns:
        str or list: Processed text or list of dialogues depending on the input type.
    """
    assert in_gb in ["CALL", "CHAT", "BOARD"], f"Invalid in_gb value: {in_gb}"

    if in_gb == "CALL":
        return preprocess_call(text)
    elif in_gb == "CHAT":
        return preprocess_chat(text)
    elif in_gb == "BOARD":
        return preprocess_couns(text)
