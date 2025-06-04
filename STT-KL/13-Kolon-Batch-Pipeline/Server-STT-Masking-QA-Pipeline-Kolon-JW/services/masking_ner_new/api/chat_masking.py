import os
import re
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    BertForTokenClassification,
    logging,
    pipeline,
)
import ahocorasick

model_name = "/workspace/static/epoch_2"

# Load the model and tokenizer with the specified offline directory
model = AutoModelForTokenClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)
ner = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# 환경 변수 재설정
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Aho-corasick algorithm
A = ahocorasick.Automaton()
# exception_name_list.txt 파일 경로
exception_name_file_path = '/workspace/static/filter_exception_list.txt'
# 파일을 읽어서 예외 처리할 이름 리스트 가져오기 (엔터로 구분된 이름들)
with open(exception_name_file_path, 'r', encoding='utf-8') as file:
    for idx, line in enumerate(file):
        A.add_word(line.strip(), (idx, line.strip()))

A.make_automaton()

def phone_masking(text):
    DIGIT_PATTERN_LIST = [
        # r"\d{10,11}", # product code 마스킹 제외하기 위해
        r"010-\d{4}-\d{4}",
        # r"\d{2,3}-\d{4}-\d{4}", # 주문번호가 마스킹되므로 해당 패턴 제외
    ]
    replace_list = []
    for pattern in DIGIT_PATTERN_LIST:
        # text = re.sub(pattern, "*", text)
        text = re.sub(pattern, "***-****-****", text)
        current_replace = {
            "preceding_text": text,
            "replace": text,
            # "start_index": number_match.start(),
            # "end_index": number_match.end(),
        }
        replace_list.append(current_replace)
        print(
            f">{current_replace['preceding_text']}\n>>{current_replace['replace']}\n\n"
        )

    # CONTEXT_PATTERN = r"(연락|전화|휴대폰)"
    # context_range = 30
    # upperbound = len(text)

    # for match in re.finditer(CONTEXT_PATTERN, text):
    #     start = max(0, match.start() - context_range)
    #     end = min(upperbound, match.end() + context_range)
    #     context = text[start:end]

    #     for pattern in DIGIT_PATTERN_LIST:
    #         for number_match in re.finditer(pattern, context):
    #             replaced_str = re.sub("\d+", "*", context)
    #             current_replace = {
    #                 "preceding_text": context,
    #                 "replace": replaced_str,
    #                 "start_index": number_match.start(),
    #                 "end_index": number_match.end(),
    #             }
    #             replace_list.append(current_replace)
    #             print(
    #                 f">{current_replace['preceding_text']}\n>>{current_replace['replace']}\n\n"
    #             )

    #     for replace_item in reversed(replace_list):
    #         text = (
    #             text[: replace_item["start_index"]]
    #             + replace_item["replace"]
    #             + text[replace_item["end_index"] :]
    #         )

    return text, replace_list


def address_masking(text):

    ADDRESS_FIND_PATTERN = [
        r"\d+\s?(동|호|층|번길|길|번지)",
        r"[가-힣]\s?\d+"  # ex) 관악대로360
        r"[가-힣]\d+대?로",  # ex) 김포한강2로
    ]

    ADDRESS_REPL_CHAR = "*"

    for pattern in ADDRESS_FIND_PATTERN:

        pattern_indices = [
            (match.start(), match.end()) for match in re.finditer(pattern, text)
        ]

        replace_list = []
        for start, end in pattern_indices:
            matching_str = text[start:end]
            replaced_str = re.sub("\d+", ADDRESS_REPL_CHAR, matching_str)
            current_replace = {
                "preceding_text": matching_str,
                "replace": replaced_str,
                "start_index": start,
                "end_index": end,
            }
            replace_list.append(current_replace)

        for replace_item in reversed(replace_list):
            text = (
                text[: replace_item["start_index"]]
                + replace_item["replace"]
                + text[replace_item["end_index"] :]
            )

    return text, replace_list


def bank_masking(text):
    CONTEXT_PATTERN_LIST = [
        r"계좌",
        r"은행",
        r"뱅크",
        r"입금",
        r"예금주",
    ]
    EXCEPT_LIST = ["주문번호", "주문 번호", "운송장"]
    upperbound = len(text)
    context_range = 50

    for pattern in CONTEXT_PATTERN_LIST:

        for match in re.finditer(pattern, text):
            start = max(0, match.start() - context_range)
            end = min(upperbound, match.end() + context_range)
            context = text[start:end]

            number_pattern = r"\d[-\d]{8,17}\d"

            if bool(re.search(number_pattern, context)):
                # If except list word appears in context, do not substitue
                break_flag = False
                for word in EXCEPT_LIST:
                    if word in context:
                        break_flag = True

                match = re.search(number_pattern, context)
                matching_str = match.group()

                # To exclude to macth 2024-07-15 (date)
                if len(re.findall(r"\d", matching_str)) < 10:
                    break_flag = True

                try:
                    # matching group의 바로 뒤 한자리
                    back = context[match.end()]
                    # To exclude to match K1717742658672050BR01 (product code)
                    if back.isdigit() or back.isalpha():
                        break_flag = True
                    # To exclude to match ~원, ~일
                    elif bool(re.match(r"(월|일|년|연|원)", back)):
                        break_flag = True
                except:
                    pass

                if not break_flag:
                    context_with_replacement, n_subs = re.subn(
                        number_pattern, "*", context
                    )
                    print(f">{context}\n>>{context_with_replacement}")
                    current_replace = {
                        "preceding_text": context,
                        "replace": context_with_replacement,
                        "start_index": start,
                        "end_index": end,
                    }
                    # print(f">{context}\n>>{context_with_replacement}\n\n")
                    text = (
                        text[: current_replace["start_index"]]
                        + current_replace["replace"]
                        + text[current_replace["end_index"] :]
                    )

    return text, []


def mask_names(text):
    # Define the regular expression patterns
    pattern1 = r"(상담사|상담원)(?!:)\s+((?:\S+\s*){1,4})(\s*입니다|이었습니다|입니다|였습니다)"
    pattern2 = r"상담사:\s*(.*?)\s*([가-힣]{2,})\s*(고객님\s*본인|고객님\s*맞)"
    long_ending_pattern = r"고객:\s*(.*?)\s*([가-힣]{2,})\s*(입니다|여|요)"
    conversation_pattern = r"(상담사: .*성함.*\n고객: .*(?:\n고객: .*)*)"
    new_pattern = r"상담사:\s*(.*?)\s*(코오롱|코오롱몰)\s*([가-힣]{2,})\s*(.*?)(이었습니다|입니다|였습니다)"
    email_pattern = r"(상담사|고객):.*?\b(\S{0,4})\s*?\S*?\s*골뱅이\s*\S*?\s*(\S{0,4})"

    def replace_first_name(match):
        if "\n" in match.group(0):
            return match.group(0)
        if any(term in match.group(2) for term in ['연결', '연락', '건가', '통화', '응답']):
            return match.group(0)
        return match.group(1) + " *** " + match.group(3)

    def replace_second_pattern(match):
        if "\n" in match.group(0):
            return match.group(0)
        if any(term in match.group(2) for term in ['분더러', '뿐더러', '애초에', '있어']):
            return match.group(0)
        return f"상담사: {match.group(1)} *** {match.group(3)}"

    def replace_customer_name_long(match):
        if "\n" in match.group(0):
            return match.group(0)
        if match.group(2).startswith(('이름', '아니', '고생', '여보세', '틀리다', '매장', '불러', '확인', '얘기')):
            return match.group(0)
        return f"고객: {match.group(1)} *** {match.group(3)}"

    def replace_kolon_name(match):
        if "\n" in match.group(0):  # 줄바꿈이 있는 경우 처리하지 않음
            return match.group(0)
        if any(term in match.group(0) for term in ["상담사", "상담원", "고객센터"]):
            return match.group(0)
        if any(term in match.group(2) for term in ['연결', '연락','응답']):
            return match.group(0)
        return f"상담사: {match.group(1)} {match.group(2)} *** {match.group(4)}{match.group(5)}"

    def mask_email(match):
        if "\n" in match.group(0):
            return match.group(0)
        return f"{match.group(1)}: {'*' * len(match.group(2).strip())}골뱅이{'*' * len(match.group(3).strip())}"

    # Apply the patterns in the order 2 -> 3 -> 1
    masked_text, count2 = re.subn(pattern2, replace_second_pattern, text)

    # Process conversation with '성함'
    def process_conversation(match):
        customer_text = match.group(0)
        masked_customer_text = []
        first_masked = False

        for customer_line in customer_text.splitlines():
            if first_masked:
                masked_customer_text.append(customer_line)
                continue
            if "\n" in customer_line:
                masked_customer_text.append(customer_line)
                continue

            new_masked_text, count_long = re.subn(long_ending_pattern, replace_customer_name_long, customer_line)
            masked_customer_text.append(new_masked_text)

            if count_long > 0:
                first_masked = True

        return "\n".join(masked_customer_text)

    masked_text, count3 = re.subn(conversation_pattern, lambda m: process_conversation(m), masked_text)

    # Apply the final pattern for 상담사 관련
    masked_text, count1 = re.subn(pattern1, replace_first_name, masked_text)


    masked_text, count_new = re.subn(new_pattern, replace_kolon_name, masked_text)


    masked_text, email_count = re.subn(email_pattern, mask_email, masked_text)

    total_count = count1 + count2 + count3 + count_new + email_count

    return masked_text, total_count


# Aho-Corasick 알고리즘을 사용하여 이름을 마스킹
def exception_name(text):
    result = list(text)

    # Aho-Corasick을 사용하여 이름 찾기
    for end_index, (idx, original_name) in A.iter(text):
        start_index = end_index - len(original_name) + 1
        
        # 이름 뒤에 특정 접미사가 있는지 검사하는 패턴
        suffix_pattern = (
            r'^(니|입|여|요|라|네|군|구|다|였|이|고|님)'  # 붙을 수 있는 접미사
        )

        # 이름 바로 뒤 한 글자만 접미사 검사
        if re.search(suffix_pattern, text[start_index + len(original_name):start_index + len(original_name) + 1]):
            result[start_index:end_index + 1] = ['*'] * len(original_name)

    return ''.join(result)

def re_masking(text):

    # Step 0: Exception name list Masking
    exception_masked_name = exception_name(text)

    # Step 1: Mask names
    masked_name, masked_name_count = mask_names(exception_masked_name)

    # Step 2: Mask addresses
    masked_address, mask_address_repl = address_masking(masked_name)

    # Step 3: Mask numbers
    masked_phone, masked_phone_repl = phone_masking(masked_address)
    masked_bank, masked_bank_rep = bank_masking(masked_phone)

    return masked_bank


def ner_masking(ner, text):
    output = ner(text)
    scores=[]
    persons = []
    for entity_list in output:
        if entity_list['entity_group']=='PS_NAME' and len(entity_list['word']) == 3 and entity_list['score'] >= 0.6 and entity_list['word'] not in ['코오롱', '더카트', '고객센','잠시만','지포어','오에로','마버니','스타필','하나스','알라린','마보니','에피그','포이트','화성구','순산구','이코롱','아노라','비에이','비케이','내시포','아니뇨','재문의','원상태','비케이','얼룩이','노이기','자켓란','비가요','구아볼','우에수','오기는','습니다','수선이','비이엘','반팔티','예주문','성트크','피티엠','오염이','하이다','리우이','정신다','미방문','현자요','네모마','방대구','마르세','슈커마','김모습','무상이','부이당','가네스','유알엘','포비스','지에스','이아이','씨제택','지제트','지티비','페이크','재연락','안타파','아트아','수수만','시구마','성수점','하스마','아티보','재회수','세리즈','이트아']:
            if not entity_list['word'].startswith('##'):
                persons.append(entity_list['word'])
                scores.append(entity_list['score'])
    mask = persons
    mask = [sublist for sublist in mask if sublist]
    # print(mask,scores)
    
    for ner_item in mask:
        text = text.replace(ner_item, f"***")
        # print(ner_item)
    return [text, mask]


def main(data, logger):
    """
    Get masked data for privacy information
    return : masked text list
    """

    # logger.info("Starting masking data for privcay information")
    result = []
    for text in data:

        try:
            re_output = re_masking(text)
        except Exception as e:
            logger.error("Ignore Regular Expression Masking by Error:" + str(e))
            logger.info("Ignore Regular Expression Masking Input:" + text)
            re_output = text

        try:
            ner_output = ner_masking(ner, re_output)
        except Exception as e:
            logger.error("Ignore NER Masking by Error:" + str(e))
            logger.info("Ignore NER Masking Input:" + re_output)

            ner_output = re_output

        result.append(ner_output[0])

    return result


# def main(text):
#     """
#     Get masked data for privacy information
#     return : masked text
#     """

#     try:
#         re_output = re_masking(text)
#     except Exception as e:
#         logger.error("Ignore Regular Expression Masking by Error:" + str(e))
#         logger.info("Ignore Regular Expression Masking Input:" + text)
#         re_output = text

#     try:
#         ner_output = ner_masking(ner, re_output)
#     except Exception as e:
#         logger.error("Ignore NER Masking by Error:" + str(e))
#         logger.info("Ignore NER Masking Input:" + re_output)

#         ner_output = [re_output]

#     result = ner_output[0]

#     return result


# if __name__ == "__main__":

#     from pororo import Pororo

#     ner = Pororo(task="ner", lang="ko")

#     path = "/home/metanet/Workspace/09-Kolon-TA/kolon_2q_voc/data/couns_chat_raw.json"
#     with open(path, "r") as f:
#         data = json.load(f)

#     logger.info(f"Data len {len(data)}")
#     # data = data[:10] # for test

#     update_data = []
#     for x in data:
#         if x["channel_type"] == "MENU_CHAT":
#             preprocessed = preprocess_chat(x["data"])
#         elif x["channel_type"] == "MENU_COUNS":
#             preprocessed = preprocess_couns(x["data"])
#         x.update(
#             {
#                 "preprocessed": preprocessed,
#                 "masked": main(preprocessed),
#             }
#         )
#         update_data.append(x)
#     print(update_data[0].keys())

#     with open(
#         "/home/metanet/Workspace/09-Kolon-TA/kolon_2q_voc/data/couns_chat_preprocessed_masked.json",
#         "w",
#         encoding="utf-8",
#     ) as f:
#         json.dump(update_data, f, ensure_ascii=False, indent=4)

#     logger.info("preprocessing, masking done.")
