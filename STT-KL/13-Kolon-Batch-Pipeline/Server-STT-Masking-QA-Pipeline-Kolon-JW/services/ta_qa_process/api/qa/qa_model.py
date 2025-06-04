import re
import pandas as pd

from openai import OpenAI
# from langchain.prompts import PromptTemplate
from functools import partial
from typing import Optional

from api.qa.qa_prompt import *


class QAModel:

    def __init__(self, client: OpenAI, model_name: str, logger=None):
        self.client = client
        self.model_name = model_name
        self.logger = logger

        # load criteria data
        path = "/workspace/api/qa/data/auto_qa_criteria.xlsx" # == v6
        df = pd.read_excel(path, sheet_name="data")
        criteria = []
        for i, row in df.iterrows():
            if row["평가 항목"] == "필수사항안내":  # 별도 프로세스
                continue
            criteria.append(
                {
                    "task_category": row["구분"],
                    "task_name": row["평가 항목"],
                    "criterion": row["criterion"],
                    "examples": row["examples"],
                    "type": row["type"],
                }
            )
        self.criteria = criteria

        df2 = pd.read_excel(path, sheet_name="필수안내사항")
        convert_dict = {}
        for i, row in df2.iterrows():
            convert_dict[str(i + 1)] = {
                "task": row["문의유형"],
                "name": row["세부사항"],
                "조건":row['조건'],
                "condition": row["condition"],
                "키워드": row["키워드"],
                "examples": row["example sentences"],
            }
        self.guidance_data = convert_dict

    def _generation(self, template, input_variables: dict, uid=None):
        """
        Generate text with OpenAI client and output text and token usage
        """

        # 코롱, 코론 -> 코오롱으로 처리
        input_variables["input"] = re.sub(
            r"코롱|코론", "코오롱", input_variables["input"]
        )
        if self.logger:
            self.logger.info(
                f"QA GPT API request input for uid {uid}: \n {input_variables['input']}"
            )

        # template = PromptTemplate.from_template(template)
        prompt = template.format(**input_variables)

        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            content = completion.choices[0].message.content
            usage = completion.usage.model_dump()

        except Exception as exc:
            print(exc)
            content = exc
            usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        return content, usage

    def assessment_func_list(self):
        funcs = [self._required_guidance]

        for data in self.criteria:
            funcs.append(partial(self._single_assessment, data))
        return funcs

    def _single_assessment(self, criterion, datapoint):
        if criterion["type"] == "P":
            output_prompt = OUTPUT_POSITIVE
        elif criterion["type"] == "N":
            output_prompt = OUTPUT_NEGATIVE

        if criterion["task_name"] == "고객 컴플레인":
            template = PROMPT_TEMPLATE_COMPLAIN
        else:
            template = PROMPT_TEMPLATE
        input_variables = {
            "instruct": INSTRUCTION,
            "assessment_type": output_prompt,
            "criterion": criterion["criterion"],
            "examples": criterion["examples"],
            "input": datapoint["MASKED_CONTENT"],
        }

        raw, usage = self._generation(template, input_variables, datapoint["UID"])
        content = self._postprocess(raw)

        # 언어습관 항목에 대한 별도 처리 (아니 그러니까요.가 상담사가 발화한게 맞는지 확인)
        if criterion["task_name"] == "언어습관":
            try:
                if content["remark"]:
                    remark = " ".join(content["remark"])
                    remark = remark.replace(" ", "")
                    if "아니그러니까요" in remark:
                        masked_content = datapoint["MASKED_CONTENT"].split("\n")
                        agent_remarks = [x for x in masked_content if "상담사" in x]
                        agent_remarks = "\n".join(
                            [x.replace(" ", "") for x in agent_remarks]
                        )
                        if "아니그러니까" not in agent_remarks:
                            if "아니그니까" not in agent_remarks:
                                content = {
                                    "assessment": "긍정",
                                    "comment": "상담사는 불필요한 발언 없이 고객의 문의에 적절히 답변하였습니다.",
                                    "remark": None,
                                }
            except Exception as e:
                self.logger.warning(f"QA-언어습관: Exception {e}")

        exist, remarks = self._validate_existance(
            datapoint["MASKED_CONTENT"], content["remark"]
        )
        if remarks:
            remarks = "/".join(remarks)

        content.update(
            {
                "exist": exist,
                "remark": remarks,
            }
        )

        result = {
            "UID": datapoint["UID"],
            "task": f"{criterion['task_category']}>{criterion['task_name']}",
            "result": {
                "content": content,
                "raw": raw,
                "usage": usage,
            },
        }

        return result

    def _required_guidance(self, datapoint):
        """필수사항안내 평가"""
        convert_dict = self.guidance_data
        inquiry_list = []
        for k, v in convert_dict.items():
            inquiry_list.append(
                f"{k}. {v['task']}>{v['name']}\nCondition:{v['condition']}"
            )
        inquiry_data = "\n\n".join(inquiry_list)

        # output format
        result = {
            "UID": datapoint["UID"],
            "task": "질문·문제해결>필수사항안내",
        }

        # 필수사항안내 15가지 문의유형 분류
        # TODO: select more than one inquiry as list
        raw1, usage1 = self._generation(
            template=REQUIREMENTS_CLS,
            input_variables={
                "data": inquiry_data,
                "input": datapoint["MASKED_CONTENT"],
            },
            uid=datapoint["UID"],
        )

        content_num = self._extract_digit(raw1)
        if content_num == "0" or content_num=="17": # 17. 온라인 취소 업체확인
            result["result"] = {
                "content": {
                    "assessment": "평가제외",
                    "remark": [],
                    "comment": "필수사항안내에 해당하는 내용이 없습니다.",
                    "exist": 0,
                },
                "raw": raw1,
                "usage": usage1,
            }
            return result

        target = convert_dict[content_num]

        raw2, usage2 = self._generation(
            template=REQUIREMENTS,
            input_variables={
                "instruct": INSTRUCTION,
                "keywords": target["키워드"],
                "examples": target["examples"],
                "input": datapoint["MASKED_CONTENT"],
            },
            uid=datapoint["UID"],
        )

        try:
            # usage = {k: v + usage2[k] for k, v in usage1.items()}
            usage_columns = {"prompt_tokens", "completion_tokens", "total_tokens"}
            usage = {}
            for col in usage_columns:
                usage[col] = usage1[col] + usage2[col]
        except:
            usage = usage1

        content = self._postprocess(raw2)
        exist, remark = self._validate_existance(
            datapoint["MASKED_CONTENT"], content["remark"]
        )
        if remark:
            remark = "/".join(remark)

        content.update(
            {
                "exist": exist,
                "remark": remark,
                "inquiry": f"{target['task']}>{target['name']}",
            }
        )

        result["result"] = {
            "content": content,
            "raw": f"{raw1}\n{raw2}",
            "usage": usage,
        }

        return result

    def _postprocess(self, raw_output):
        """Postprocess raw model output into dict"""

        # parse json
        json_obj = self._safe_eval(raw_output)

        # convert ENG assessment into KOR
        assess = json_obj["assessment"].lower()
        if "positive" in assess:
            assess = "긍정"
        elif "negative" in assess:
            assess = "부정"
        else:
            assess = "ERROR"
        json_obj["assessment"] = assess

        # unify remark type into string
        remark = json_obj["remark"]
        if "none" in raw_output.lower():
            remark = None
        elif remark == "":
            remark = None
        elif type(remark) == list:
            if len(remark) == 0:
                remark = None
            # else:
            #     remark = "/".join(json_obj["remark"])
        json_obj["remark"] = remark

        return json_obj

    def _safe_eval(self, raw_output: str) -> dict:
        """
        Convert raw output into json object

        "```json {}```"
        """

        # remove double bracket {{}}
        start_idx = [match.start() for match in re.finditer(r"{", raw_output)][0]
        end_idx = [match.start() for match in re.finditer(r"}", raw_output)][-1]
        if raw_output[start_idx + 1] == "{":
            start_idx += 1
        if raw_output[end_idx - 1] == "}":
            end_idx -= 1
        text = raw_output[start_idx : end_idx + 1]
        text = text.replace("\n", "")

        # json_obj = eval(text)

        # cleaning quotes
        text = re.sub("\s+", " ", text)
        indices_to_remove = set()
        length = len(text)

        for i, char in enumerate(text):
            if char in {'"', "'"}:
                check_indices = list(range(max(0, i - 2), min(length, i + 3)))
                check_indices.remove(i)

                if not any(
                    text[idx] in {"{", "}", ":", ",", "[", "]"} for idx in check_indices
                ):
                    indices_to_remove.add(i)

        cleaned_text = "".join(
            [char for i, char in enumerate(text) if i not in indices_to_remove]
        )

        try:
            cleaned_text = cleaned_text.replace("null", "None")
        except Exception as e:
            self.logger.warning(f"QA-safe_eval() Exception: {e}")


        # json_obj = eval(cleaned_text)
        try:
            json_obj = eval(cleaned_text)
        except:
            try:
                if 'Negative' in cleaned_text:
                    asse = 'Negative'
                elif 'Positive' in cleaned_text:
                    asse = 'Positive'
                remark = cleaned_text.find('remark')+6
                comment = cleaned_text.find('comment')
                remark = cleaned_text[remark:comment]
                if 'None' in remark:
                    remark = None
                else:
                    remark = re.sub(r"\'\":,", '', remark)
                comment = cleaned_text[comment+7:]
                if 'None' in remark:
                    comment = None
                else:
                    comment = re.sub(r"\'\"\:{}", '', comment)
                json_obj = {
                    'assessment': asse,
                    'remark':remark,
                    'comment':comment,
                }
            except Exception as e:
                pass

        return json_obj

    def _extract_digit(self, text):
        """extract integer from string"""
        number = re.search(r"\d+", text).group()
        return number

    def _validate_existance(
        self,
        input_text,
        remarks: Optional[list],  # list or None
    ):
        """Check if generated remarks exist in input text"""

        input_text = re.sub(r"\s+", "", input_text)

        if remarks:
            exists = []
            for x in remarks:
                x = re.sub(r".,?!", "", x)
                x = re.sub(r"\s+", "", x)
                if x not in input_text:
                    exists.append(False)
                else:
                    exists.append(True)
            if all(exists):
                exists = 1
            elif any(exists):
                remarks = [remarks[i] for i, x in enumerate(exists) if x]
                exists = 1
            else:
                exists = 0
        else:
            exists = 1

        return exists, remarks
