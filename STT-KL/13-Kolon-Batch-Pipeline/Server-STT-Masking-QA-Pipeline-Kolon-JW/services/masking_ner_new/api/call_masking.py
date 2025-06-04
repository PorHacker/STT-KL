import re
import os
from transformers import AutoTokenizer, AutoModelForTokenClassification, BertForTokenClassification, logging, pipeline
import ahocorasick

class AhoCorasickNode:
    def __init__(self):
        self.children = {}
        self.fail_link = None
        self.output = []

class AhoCorasickAutomaton:
    def __init__(self):
        self.root = AhoCorasickNode()

    def add_word(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = AhoCorasickNode()
            node = node.children[char]
        node.output.append(word)

    def build_fail_links(self):
        # BFS로 Fail 링크 설정
        queue = []
        for node in self.root.children.values():
            node.fail_link = self.root
            queue.append(node)

        while queue:
            current_node = queue.pop(0)
            for char, child_node in current_node.children.items():
                # Fail 링크 설정
                fail_node = current_node.fail_link
                while fail_node is not None and char not in fail_node.children:
                    fail_node = fail_node.fail_link
                child_node.fail_link = fail_node.children[char] if fail_node else self.root
                # Output 병합
                if child_node.fail_link:
                    child_node.output += child_node.fail_link.output
                queue.append(child_node)

    def search(self, text):
        node = self.root
        results = []
        for i, char in enumerate(text):
            while node is not None and char not in node.children:
                node = node.fail_link
            if node is None:
                node = self.root
                continue
            node = node.children[char]
            if node.output:
                for pattern in node.output:
                    start_index = i - len(pattern) + 1
                    results.append((start_index, i, pattern))
        return results

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

masking_except_list = [
    "배송",
    "주문번호",
    "주문 번호" "영업일",
    "영업 일",
    "사이즈",
    "검수기간",
    "검수 기간",
    "결제수단",
    "결제 수단",
    "환불",
    "회수기사",
    "회수 기사",
    "접수번호",
    "접수 번호" "구매일로부터",  #  add "구매일로부터" instead of "지원"
    "수선" "브랜드" "금액",
    "송장",
    "신청번호",
    "신청 번호",
]


def telephone_masking(text, word_range=100):
    """전화번호 마스킹 코드
    STT 결과 text는 한자리 숫자의 경우 띄어쓰기가 있습니다. ex_ 일 이 삼 사 공 일 공
    공 일 공|공 일공으로 전화 번호 패턴을 인식합니다.
    패턴 리스트에는 패턴이 5개 있고, 각 패턴 별 대체할 마스킹 텍스트는 TELEPHONE_REPL_CHAR_LIST에 있습니다. ex_ 1번 패턴에 걸리면 *1 으로 텍스트 대체
    1번째 패턴 : 년월달분원이 숫자 뒤에 오지 않고(단위가 아닌 전화번호만 정확히 인식), 숫자 앞뒤로 띄어쓰기가 있으면 패턴 인식
    2번째 패턴 : 숫자 앞에 띄어쓰기 없고(문장 내 첫 시작 텍스트), 뒤에 띄어쓰기가 있는 경우 패턴
    3번째 패턴 : 숫자 앞에 띄어쓰기, 뒤에 \n 개행문자
    4번째 패턴 : 숫자 앞에 띄어쓰기, 뒤에 '에'라는 문자가 오는 경우 패턴
    5번째 패턴 : 숫자 앞에 띄어쓰기, 뒤에 '이요|요'라는 문자가 오는 경우 패턴
    """

    TELEPHONE_FIND_PATTERN = r"공 일 공|공 일공"
    TELEPHONE_REPL_PATTERN_LIST = [
        r"(?<=[ ])[공일이삼사오육칠팔구영]+(?=[ ](?!([년월달분원])))|(?<=[ ])하나+[ ]|(?<=[ ])둘+[ ]",  # Replace number with space before and after
        r"^[공일이삼사오육칠팔구영]+[ ]|^하나+[ ]|^둘+[ ]",  # At begining and followed with one space
        r"(?<=[ ])[공일이삼사오육칠팔구영]+[\n]|하나+[\n]|둘+[\n]",  # Between space and new-line
        r"(?<=[ ])[공일이삼사오육칠팔구영]+[에]|하나+[에]|둘+[에]",  # Between space and 에
        r"(?<=[ ])[공일이삼사오육칠팔구영]+(?:이요|요)|하나+(?:이요|요)|둘+(?:이요|요)",  # Between space and 이요/요
    ]
    TELEPHONE_REPL_CHAR_LIST = ["* ", "* ", "*\n", "*에", "*요"]

    MIN_SUB_COUNT = 2
    # Find all occurrences of the pattern
    pattern_indices = [
        match.start() for match in re.finditer(TELEPHONE_FIND_PATTERN, text)
    ]

    # Replace the pattern within the specified word range

    replaced_list = []
    for idx, start_index in enumerate(pattern_indices):
        current_replace = {}

        upperbound = len(text)
        if idx + 1 < len(pattern_indices):
            # Update upperbound
            upperbound = pattern_indices[idx + 1]

        # Extract a word_range-word context around the pattern
        start_index = max(start_index, 0)
        end_index = min(start_index + word_range, upperbound)

        context_around_pattern = text[start_index:end_index]

        # Check if "감사합니다" is in the context and update end_index
        thanks_index = context_around_pattern.find("감사합니다")
        if thanks_index != -1:
            end_index = start_index + thanks_index
            context_around_pattern = text[start_index:end_index]

        # Add context use for except list only
        except_start_index = max(start_index - 10, 0)
        except_context_around_pattern = text[except_start_index:end_index]

        # Use regular expression to replace the pattern with the replacement text
        # context_around_pattern = context_around_pattern.replace("
        context_with_replacement = context_around_pattern
        n_subs_count = 0
        for replace_pattern, replace_word in zip(
            TELEPHONE_REPL_PATTERN_LIST, TELEPHONE_REPL_CHAR_LIST
        ):
            context_with_replacement, n_subs = re.subn(
                replace_pattern,
                replace_word,
                context_with_replacement,
                count=0,
                flags=0,
            )
            n_subs_count += n_subs
        current_replace["context"] = context_around_pattern
        current_replace["except_context"] = context_with_replacement
        current_replace["replace"] = context_with_replacement
        current_replace["start_index"] = start_index
        current_replace["end_index"] = end_index
        current_replace["n_subs"] = n_subs_count
        replaced_list.append(current_replace)

    # # Replace the context in the original text
    # for replace_item in reversed(replaced_list):
    #     text = text[:replace_item["start_index"]] + replace_item["replace"] + text[replace_item["end_index"]:]

    ## 2024.07.12 Add masking except list:

    # Replace the context in the original text
    for replace_item in reversed(replaced_list):
        if replace_item["n_subs"] >= MIN_SUB_COUNT:
            # Collect items in masking_except_list that are found in replace_item["context"]
            except_found_items = [
                item
                for item in masking_except_list
                if item in replace_item["except_context"]
            ]
            replace_item["except_found"] = except_found_items

            # Check if none of the items in masking_except_list are in the context
            if not except_found_items:
                text = (
                    text[: replace_item["start_index"]]
                    + replace_item["replace"]
                    + text[replace_item["end_index"] :]
                )
                replace_item["is_rep"] = True
            else:
                replace_item["is_rep"] = False
        else:
            replace_item["is_rep"] = False

    return text, replaced_list


def number_masking(text, word_range=100):
    PRIV_NUM_FIND_PATTERN = r"연락처|번호|카드|팩스|은행|계좌"  # 계좌추가
    PRIV_NUM_REPL_PATTERN_LIST = [
        r"(?<=[ ])[공일이삼사오육칠팔구영]+(?=[ ](?!([년월달분원])))|(?<=[ ])하나+[ ]|(?<=[ ])둘+[ ]",  # Replace number with space before and after
        r"^[공일이삼사오육칠팔구영]+[ ]|^하나+[ ]|^둘+[ ]",  # At begining and followed with one space
        r"(?<=[ ])[공일이삼사오육칠팔구영]+[\n]|하나+[\n]|둘+[\n]",  # Between space and new-line
        r"(?<=[ ])[공일이삼사오육칠팔구영]+[에]|하나+[에]|둘+[에]",  # Between space and 에
        r"(?<=[ ])[공일이삼사오육칠팔구영]+(?:이요|요)|하나+(?:이요|요)|둘+(?:이요|요)",  # Between space and 이요/요
    ]
    PRIV_NUM_REPL_CHAR_LIST = ["* ", "* ", "*\n", "*에", "*요"]
    MIN_SUB_COUNT = 3

    # Find all occurrences of the pattern
    pattern_indices = [
        match.start() for match in re.finditer(PRIV_NUM_FIND_PATTERN, text)
    ]

    # Replace the pattern within the specified word range

    replaced_list = []
    for idx, start_index in enumerate(pattern_indices):
        current_replace = {}

        upperbound = len(text)
        if idx + 1 < len(pattern_indices):
            # Update upperbound
            upperbound = pattern_indices[idx + 1]

        # Extract a word_range-word context around the pattern
        start_index = max(start_index, 0)
        end_index = min(start_index + word_range, upperbound)

        context_around_pattern = text[start_index:end_index]

        # Check if "감사합니다" is in the context and update end_index
        thanks_index = context_around_pattern.find("감사합니다")
        if thanks_index != -1:
            end_index = start_index + thanks_index
            context_around_pattern = text[start_index:end_index]

        # Add context use for except list only
        except_start_index = max(start_index - 10, 0)
        except_context_around_pattern = text[except_start_index:end_index]

        # Use regular expression to replace the pattern with the replacement text
        # context_around_pattern = context_around_pattern.replace("
        context_with_replacement = context_around_pattern
        n_subs_count = 0
        for replace_pattern, replace_word in zip(
            PRIV_NUM_REPL_PATTERN_LIST, PRIV_NUM_REPL_CHAR_LIST
        ):
            context_with_replacement, n_subs = re.subn(
                replace_pattern,
                replace_word,
                context_with_replacement,
                count=0,
                flags=0,
            )
            n_subs_count += n_subs
        current_replace["context"] = context_around_pattern
        current_replace["except_context"] = context_with_replacement
        current_replace["replace"] = context_with_replacement
        current_replace["start_index"] = start_index
        current_replace["end_index"] = end_index
        current_replace["n_subs"] = n_subs_count
        replaced_list.append(current_replace)

    # Replace the context in the original text
    for replace_item in reversed(replaced_list):
        if replace_item["n_subs"] >= MIN_SUB_COUNT:
            # Collect items in masking_except_list that are found in replace_item["context"]
            except_found_items = [
                item
                for item in masking_except_list
                if item in replace_item["except_context"]
            ]
            replace_item["except_found"] = except_found_items

            # Check if none of the items in masking_except_list are in the context
            if not except_found_items:
                text = (
                    text[: replace_item["start_index"]]
                    + replace_item["replace"]
                    + text[replace_item["end_index"] :]
                )
                replace_item["is_rep"] = True
            else:
                replace_item["is_rep"] = False

        else:
            replace_item["is_rep"] = False

    return text, replaced_list


def birthday_masking(text, word_range=100):
    """생년월일 마스킹 코드
    생년월일|생년월으로 생년월일 패턴을 인식합니다.
    BIRTHDAY_REPL_PATTERN_LIST(패턴 리스트)에 패턴 1개만 존재, 패턴이 인식 되면 텍스트가 *1 "로 대체됩니다.
    (천|이천)? : 맨 앞에 천 또는 이천이 있어도 되고, 없어도 됩니다. (생년월일 첫번째에 천구십구년 등 말할 때)
    천|백|십[일이삼사오육칠팔구]| : 정확히 천, 백을 인식하기 위함
    """
    BIRTHDAY_FIND_PATTERN = r"생년월일|생년월"
    BIRTHDAY_REPL_PATTERN_LIST = [
        r"\b(?:(천|이천)?[영공일이삼사오육칠팔구십유시]+[천백십]?[일이삼사오육칠팔구]*[십]?[일이삼사오육칠팔구]*[년월일]*|천|백|십[일이삼사오육칠팔구]|(?<=[ ])하나|(?<=[ ])둘|[영공일이삼사오육칠팔구십유시]+[년월일]+(이요|요)*)\b",
    ]

    BIRTHDAY_REPL_CHAR_LIST = ["* ", "* ", "*\n", "*이요", "*", "*", "*", "*"]

    # Find all occurrences of the pattern
    pattern_indices = [
        match.start() for match in re.finditer(BIRTHDAY_FIND_PATTERN, text)
    ]

    # Replace the pattern within the specified word range
    replaced_list = []
    for idx, start_index in enumerate(pattern_indices):
        current_replace = {}

        upperbound = len(text)
        if idx + 1 < len(pattern_indices):
            # Update upperbound
            upperbound = pattern_indices[idx + 1]

        # Extract a word_range-word context around the pattern
        start_index = max(start_index, 0)
        end_index = min(start_index + word_range, upperbound)

        context_around_pattern = text[start_index:end_index]

        # Check if "감사합니다" is in the context and update end_index
        thanks_index = context_around_pattern.find("감사합니다")
        if thanks_index != -1:
            end_index = start_index + thanks_index
            context_around_pattern = text[start_index:end_index]

        # Use regular expression to replace the pattern with the replacement text
        context_with_replacement = context_around_pattern
        n_subs_count = 0
        for replace_pattern, replace_word in zip(
            BIRTHDAY_REPL_PATTERN_LIST, BIRTHDAY_REPL_CHAR_LIST
        ):
            context_with_replacement, n_subs = re.subn(
                replace_pattern,
                replace_word,
                context_with_replacement,
                count=0,
                flags=0,
            )
            n_subs_count += n_subs
        current_replace["context"] = context_around_pattern
        current_replace["replace"] = context_with_replacement
        current_replace["start_index"] = start_index
        current_replace["end_index"] = end_index
        current_replace["n_subs"] = n_subs_count

        replaced_list.append(current_replace)

    # Replace the context in the original text
    for replace_item in reversed(replaced_list):
        text = (
            text[: replace_item["start_index"]]
            + replace_item["replace"]
            + text[replace_item["end_index"] :]
        )

    return text, replaced_list


def address_masking(text, word_range=30, preceding_search_range=100):

    # Construct the find pattern to match any of the defined patterns

    ADDRESS_FIND_PATTERN = (
        r"\b"  # Word boundary
        r"[천백십]?[일이삼사오육칠팔구]+"  # Korean number pattern (1 to 9999)
        r" (호|층)"  # Match either "호" or "층"
        r"(?:이[a-zA-Z가-힣]+|시[a-zA-Z가-힣]+)?"  # Optional: Matches "이" or "시" followed by any sequence of Korean characters or English letters
        r"(?:\b|로|에|에서|이라고|이요|인가요|이|가|여|요|이시구요|예요|라고|거든요)"  # Non-capturing group for various word endings
        r"\b"  # Word boundary
    )

    ADDRESS_REPL_PATTERN = [
        r"\b(?:(천|이천)?[영공일이삼사오육칠팔구십]+[천백십]?[일이삼사오육칠팔구]*[십]?[일이삼사오육칠팔구]*[호|동|층]*|천|백|십[일이삼사오육칠팔구]|(?<=[ ])하나|(?<=[ ])둘|[영공일이삼사오육칠팔구십]+[호|동|층]+(이요|요|이시구요|예요|라고|거든요)*)\b"
    ]
    ADDRESS_REPL_CHAR = ["*"]
    CONTEXT_WORDS = r"(?:주소|아파트|빌라|빌딩|마을|아이파크|배송지|장소|건물|동)"

    # Find all occurrences of the pattern
    pattern_indices = [match.end() for match in re.finditer(ADDRESS_FIND_PATTERN, text)]

    # Replace the pattern within the specified word range
    replaced_list = []
    for idx, end_index in enumerate(pattern_indices):
        current_replace = {}

        lowerbound = 0
        if idx > 0:
            # Update lowerbound
            lowerbound = pattern_indices[idx - 1]

        # Extract a word_range-word context around the pattern
        start_index = max(end_index - word_range, lowerbound)

        # Get the preceding text within the word range
        preceding_text = text[max(0, start_index - preceding_search_range) : end_index]

        context_around_pattern = text[start_index:end_index]

        ## Find and update the start_index if specific words occurr in the context
        # Find all occurrences of specific words (stop words) in the context
        specific_word_pattern = r"(?:빌딩|빌라|번길|동|동에|길|아파트|:)"
        specific_word_indices = [
            m.start()
            for m in re.finditer(specific_word_pattern, context_around_pattern)
        ]
        # Check if context_around_pattern contains ":"
        if specific_word_indices:
            last_index = specific_word_indices[-1]
            start_index += last_index  # Update start_index to the position of the last ":" occurrence
            context_around_pattern = text[
                start_index:end_index
            ]  # Update context_around_pattern

        # Check if one of the context words appears in the preceding text
        if re.search(CONTEXT_WORDS, preceding_text):
            # Use regular expression to replace the pattern with the replacement text
            context_with_replacement = context_around_pattern
            n_subs_count = 0
            for replace_pattern, replace_word in zip(
                ADDRESS_REPL_PATTERN, ADDRESS_REPL_CHAR
            ):
                context_with_replacement, n_subs = re.subn(
                    replace_pattern,
                    replace_word,
                    context_with_replacement,
                    count=0,
                    flags=0,
                )
                n_subs_count += n_subs
            current_replace["preceding_text"] = preceding_text
            current_replace["context"] = context_around_pattern
            current_replace["replace"] = context_with_replacement
            current_replace["start_index"] = start_index
            current_replace["end_index"] = end_index
            current_replace["n_subs"] = n_subs_count

            replaced_list.append(current_replace)

    # Replace the context in the original text
    for replace_item in reversed(replaced_list):
        text = (
            text[: replace_item["start_index"]]
            + replace_item["replace"]
            + text[replace_item["end_index"] :]
        )

    return text, replaced_list


def mask_names(text):
    # Define the regular expression patterns
    pattern1 = r"(상담사|상담원|고객센터)(?!:)\s+((?:\S+\s*){1,4})(\s*입니다|이었습니다|입니다|였습니다)"
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

def re_masking(text, word_range=100):

    # Step 0: Exception name list Masking
    exception_masked_name = exception_name(text)

    # Step 1: Mask names
    masked_name, masked_name_count = mask_names(exception_masked_name)

    # Step 2: Mask addresses
    masked_address, mask_address_repl = address_masking(masked_name, word_range=50)

    # Step 3: Mask telephone numbers
    masked_telephone, masked_telephone_repl = telephone_masking(masked_address, 70)

    # Step 4: Mask numbers
    masked_number, masked_number_repl = number_masking(masked_telephone, word_range)

    # Step 5: Mask birthdays
    masked_birthday, masked_birthday_repl = birthday_masking(masked_number, word_range)

    return masked_birthday


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


# if __name__ == "__main__":
#     data= ["상담원 이었습니다. 전화번호는 공 일 공 일공 입니다. 생년월일은 이천공일년 공이월 공이일입니다. 주소는 서울시 강남구 역삼동 삼호 입니다. 감사합니다."]
#     result = main(data)
#     print(result)
