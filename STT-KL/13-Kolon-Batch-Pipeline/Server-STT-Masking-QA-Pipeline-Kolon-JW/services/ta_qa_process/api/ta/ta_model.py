import re
import pymysql
import pandas as pd
import datetime
import numpy as np
from collections import defaultdict
from openai import OpenAI
# from langchain.prompts import PromptTemplate
from api.ta.ta_prompt import *

class VoCModel:
    """
    TA model for single or batch input

    Prior tasks :
    - category classification
    - brand name extraction
    - speaker summary

    Subsequent tasks :
    - title generation
    - content summary
    - sentiment classification
    - keyword extraction

    2024.09.04 fix :
    Run category classification, sentiment classification in one func
    """

    def __init__(self, openai_client, model_name, category, logger=None):

        self.client = openai_client
        self.model_name = model_name
        self.logger = logger

        if category is not None:
            d2_info = {}    
            self.d2 = {}
            # d3_info = defaultdict(list)
            self.d3 = defaultdict(list)
            for x in category: # 1, 3, 6
                d2_info[x[3]] = {'d1':x[1], 'desc':x[4].strip()}
                self.d3[x[3]].append({
                    'd3_name':x[6],
                    'd3_desc':x[7].strip()
                })
            for i, (k, v) in enumerate(d2_info.items()):
                self.d2[str(i+1)] = {
                    'd2_name':k,
                    'd2_desc':v['desc'],
                    'd1':v['d1']
                }

        # data for category classification
        # path = "/workspace/api/data/코오롱_상담유형.xlsx"
        # path = "/workspace/api/data/ta_category.xlsx"
        # self.df = pd.read_excel(path)

        # category = self._call_db_procedure()
        # d2_info = {}
        # self.d2 = {}
        # # d3_info = defaultdict(list)
        # self.d3 = defaultdict(list)
        # for x in category: # 1, 3, 6
        #     d2_info[x[3]] = {'d1':x[1], 'desc':x[4].strip()}
        #     self.d3[x[3]].append({
        #         'd3_name':x[6],
        #         'd3_desc':x[7].strip()
        #     })
        # for i, (k, v) in enumerate(d2_info.items()):
        #     self.d2[str(i+1)] = {
        #         'd2_name':k,
        #         'd2_desc':v['desc'],
        #         'd1':v['d1']
        #     }
        
        # data for brand name extraction
        # brand_list_path = "/workspace/api/data/kolon_brand_list_updated.xlsx"  # file updated 2024.08.19 (WAAC->왁)
        brand_list_path = "/workspace/api/data/kolon_brand_list_updated.xlsx"
        brand_list = pd.read_excel(brand_list_path)
        brand_info = []
        for i, row in brand_list.iterrows():
            brand_info.append((row["브랜드명"], row["브랜드영문명"]))
        brand_info = brand_info[:-1]
        # brand_list = brand_list["브랜드명"].tolist()[:-1]  # 기타 제외
        self.brand_dict = {str(i + 1): x for i, x in enumerate(brand_info)}
        self.brand_prompt = "\n".join(
            [f"{i}. {v[0]}({v[1]})" for i, v in self.brand_dict.items()]
        )

    def generate(self, template, input_variables: dict, logprob=False, uid=None):
        input_variables["instruct"] = INSTRUCTION
        prompt = template.format(**input_variables)
        # print(prompt)

        model_params = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": prompt},
            ],
            "logprobs": logprob,
        }

        # max context length error
        try:
            completion = self.client.chat.completions.create(**model_params)


            content = completion.choices[0].message.content
            # log GPT API request parameters
            if self.logger:
                self.logger.info(
                    f"GPT API request context for UID {uid}:\n{input_variables['input']}\nRaw Output:\n{content}"
                )
            usage = (
                completion.usage.model_dump()
            )  # prompt_tokens, completion_tokens, total_prompts
            if logprob:
                # TODO: find number index and use that position logprob
                logprob_value = completion.choices[0].logprobs.content[0].logprob
                # convert lobprob into linear prob for convenience with exponential
                linear_logprob = np.round(np.exp(logprob_value) * 100, 10)
                # print(content, linear_logprob)
                usage.update(
                    {"logprob": logprob_value, "linear_logprob": linear_logprob}
                )

        except Exception as e:
            print(e)
            content = f"ERROR:{str(e)}"
            usage = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "logprob": 0,
                "linear_logprob": 0,
            }

        return content, usage

    def category_classification(self, datapoint):
        """
        Add returning logprobs for classification
        2024.10.08 update : using db procedure to get mapping
        """
        input_text = datapoint["MASKED_CONTENT"]

        # df = self.df
        # d1_map = {}  # d2 to d1
        # d2_info = defaultdict(dict)
        # d1 = df.iloc[0]["대분류"]
        # d2 = df.iloc[0]["중분류"]
        # for i, row in df.iterrows():
        #     cur_d1 = row["대분류"]
        #     cur_d2 = row["중분류"]
        #     if pd.isnull(cur_d1):
        #         cur_d1 = d1
        #     else:
        #         d1 = cur_d1
        #     if pd.isnull(cur_d2):
        #         cur_d2 = d2
        #         d2_info[d2]["d3"][row["소분류"]] = row["소분류_desc"]
        #     else:
        #         d2 = cur_d2
        #         d2_info[cur_d2] = {
        #             "d2_desc": row["중분류_desc"],
        #             "d3": {
        #                 row["소분류"]: row["소분류_desc"],
        #             },
        #         }
        #     d1_map[cur_d2] = cur_d1

        # 중분류
        # d2_prompt = []
        # d2_map = {}
        # for i, item in enumerate(d2_info.items()):
        #     k, v = item
        #     d2_prompt.append(f"{i+1}. {k} ({v['d2_desc']})")
        #     d2_map[str(i + 1)] = k

        d2_prompt = "\n".join([f"{k}. {v['d2_name']}: {v['d2_desc']}" for k, v in self.d2.items()])

        template = CATEGORY_CLS
        input_variables = {
            "input": input_text,
            "data": d2_prompt,
        }
        raw1, usage1 = self.generate(
            template, input_variables, logprob=True, uid=datapoint["UID"]
        )

        d2_key_number = str(self._extract_digit(raw1))
        # d2_category = d2_map[d2_key_number]
        d2_category = self.d2[d2_key_number]['d2_name']
        d1_category = self.d2[d2_key_number]['d1']

        # 소분류
        d3 = self.d3[d2_category]
        d3_map = {str(i+1):x['d3_name'] for i, x in enumerate(d3)}
        d3_prompt = "\n".join([f"{i+1}. {x['d3_name']}({x['d3_desc']})" for i, x in enumerate(d3)])
        # d3_info = d2_info[d2_category]["d3"]
        # d3_prompt = []
        # d3_map = {}
        # for i, item in enumerate(d3_info.items()):
        #     k, v = item
        #     d3_prompt.append(f"{i+1}. {k} ({v})")
        #     d3_map[str(i + 1)] = k

        template = CATEGORY_CLS
        input_variables = {
            "input": input_text,
            "data": d3_prompt
        }
        raw2, usage2 = self.generate(
            template, input_variables, logprob=True, uid=datapoint["UID"]
        )

        d3_key_number = str(self._extract_digit(raw2))
        d3_category = d3_map[d3_key_number]

        try:
            # usage = {k: v + usage2[k] for k, v in usage1.items()}
            usage_columns = ["prompt_tokens", "completion_tokens", "total_tokens"]
            usage = {}
            for col in usage_columns:
                usage[col] = usage1[col] + usage2[col]
        except Exception as e:
            print(e)
            usage = usage1

        result = {
            "UID": datapoint["UID"],
            "task": "category_classification",
            "result": {
                # "content": f"{d2_category}>{d3_category}", # save d2, d3 seperately in db
                "content": {
                    "depth1": d1_category,
                    "depth1_prob": usage1["linear_logprob"], # use same prob of depth2
                    "depth2": d2_category,
                    "depth2_prob": usage1["linear_logprob"],
                    "depth3": d3_category,
                    "depth3_prob": usage2["linear_logprob"],
                },
                "usage": usage,
                "raw": "\n".join([raw1, raw2]),
            },
        }

        return result

    def brand_extraction(self, datapoint):
        """
        Extract mentioned brand name from given brand list.
        """
        input_text = datapoint["MASKED_CONTENT"]

        template = BRAND_EXTRACT
        input_variables = {"input": input_text, "data": self.brand_prompt}
        raw, usage = self.generate(template, input_variables, uid=datapoint["UID"])

        # parsing
        if "none" in raw.lower():
            content = "None"
        else:
            if re.match(r"\d+", raw):
                brand_num = self._extract_digit(raw)
                try:
                    content = self.brand_dict[brand_num][0]
                except KeyError:
                    content = "None"
            else:
                content = "None"  # 브랜드명 리스트에 있는 것만 사용.
            if content == "코오롱스포츠":
                if "스포츠" not in input_text and "sport" not in input_text.lower():
                    content = "None"

        result = {
            "UID": datapoint["UID"],
            "task": "brand_extraction",
            "result": {
                "content": content,
                "raw": raw,
                "usage": usage,
            },
        }
        return result

    def speaker_summarization(self, datapoint):
        """
        Generate speaker specific summary for agent, customer.
        default summary when input length under 20
        content : {
            "customer": (customer summary),
            "agent": (agent summary)
        }
        """
        input_text = datapoint["MASKED_CONTENT"]

        # 고객이 발화한 텍스트 길이가 일정 이하일 때 고객 발화 없음으로 판단
        # try:
        #     customer = []
        #     text_list = input_text.split("\n")
        #     customer_flag = None
        #     for x in text_list:
        #         if "고객:" in x:
        #             customer.append(x[3:])
        #             customer_flag = True
        #         elif "상담사:" in x:
        #             customer_flag = False
        #         else:
        #             if customer_flag:
        #                 customer.append(x)
        #     customer = ''.join(customer)
        #     if len(customer) <= 1:
        #         result = {
        #             "UID": datapoint["UID"],
        #             "task": "speaker_summarization",
        #             "result": {
        #                 "content": {
        #                     "customer":"고객 발화 없음",
        #                     "agent":"고객의 발화 없이 무응답 종결",
        #                 },
        #                 "raw": "",
        #                 "usage": {
        #                     "prompt_tokens":0,
        #                     "completion_tokens":0,
        #                     "total_tokens":0,
        #                 },
        #             },
        #         }
        #         return result
        # except Exception as e:
        #     self.logger.warning(f"TA speaker summary {e}")

        template = SPEAKER_SPECIFIC_SUMMARY
        input_variables = {
            "input": input_text,
        }
        raw, usage = self.generate(
            template=template, input_variables=input_variables, uid=datapoint["UID"]
        )

        content = self._json_parse(raw)
        #241002 요청사항 반영
        #content["customer"] = content["customer"].replace("고객은", "")
        #content["agent"] = content["agent"].replace("상담사는", "")
        content["customer"] = re.sub(r"고객[은는이가]", "", content["customer"])
        content["agent"] = re.sub(r"상담사[은는이가]", "", content["agent"])
        result = {
            "UID": datapoint["UID"],
            "task": "speaker_summarization",
            "result": {
                "content": content,
                "raw": raw,
                "usage": usage,
            },
        }
        return result

    def title_generation(self, datapoint):
        input_text = datapoint["MASKED_CONTENT"]

        template = TITLE_GENERATION
        input_variables = {"input": input_text}
        raw, usage = self.generate(template, input_variables, uid=datapoint["UID"])

        content = self._json_parse(raw)["title"]
        content = content.replace("*", "")

        result = {
            "UID": datapoint["UID"],
            "task": "title_generation",
            "result": {
                "content": content,
                "raw": raw,
                "usage": usage,
            },
        }
        return result

    def content_summarization(self, datapoint):
        """전체 상담텍스트 요약"""
        input_text = datapoint["MASKED_CONTENT"]

        prompt = CONVERSATION_SUMMARY
        input_variables = {
            "input": input_text,
        }

        raw, usage = self.generate(prompt=prompt, input_variables=input_variables)

        content = self._json_parse(raw)["summary"]
        result = {
            "UID": datapoint["UID"],
            "task": "content_summarization",
            "result": {
                "content": content,
                "raw": raw,
                "usage": usage,
            },
        }
        return result

    def sentiment_classification(self, datapoint):
        """고객의 상담결과에 대한 감정 긍,부정,중립으로 분류"""

        input_text = datapoint["MASKED_CONTENT"]

        template = CUSTOMER_SENTIMENT_CLS
        input_variables = {
            "input": input_text,
        }

        raw, usage = self.generate(
            template=template, input_variables=input_variables, uid=datapoint["UID"]
        )
        if "positive" in raw.lower():
            content = "긍정"
        elif "neutral" in raw.lower():
            content = "중립"
        elif "negative" in raw.lower():
            content = "부정"
        else:
            raise Exception(f"Can't convert sentiment_classification output : {raw}")

        result = {
            "UID": datapoint["UID"],
            "task": "sentiment_classification",
            "result": {
                "content": content,
                "raw": raw,
                "usage": usage,
            },
        }
        return result

    def keyword_extraction(self, datapoint):
        """키워드 3개 추출"""

        input_text = datapoint["MASKED_CONTENT"]

        template = KEYWORD_EXTRACT
        input_variables = {
            "input": input_text,
        }

        raw, usage = self.generate(
            template=template, input_variables=input_variables, uid=datapoint["UID"]
        )

        content = self._json_parse(raw)
        content = list(content.values())
        try:
            content = [x.replace("*", "") for x in content]
            content = [x for x in content if len(x) > 0]
        except Exception as e:
            self.logger.warning(
                f"TA-keyword extraction : content result {content}, Exception {e}"
            )

        if len(content) > 3:
            content = content[:3]

        result = {
            "UID": datapoint["UID"],
            "task": "keyword_extraction",
            "result": {
                "content": content,
                "raw": raw,
                "usage": usage,
            },
        }
        return result

    def category_sentiment(self, datapoint):
        """
        소분류 텍스트에 불만, 만족이 있는 경우 감정분류 없이 감정을 각각 부정, 긍정으로 찍기
        """

        category_cls_result = self.category_classification(datapoint)
        depth3_val = category_cls_result["result"]["content"]["depth3"]

        if "불만" in depth3_val:
            self.logger.info(f"UID {datapoint['UID']} category cls result '{depth3_val}' -> returning sentiment cls as '부정' manually")
            sentiment_result = {
                "UID": datapoint["UID"],
                "task": "sentiment_classification",
                "result": {
                    "content": "부정",
                    "raw": "소분류 불만 -> 감정분류 부정",
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    },
                },
            }
        elif "만족" in depth3_val:
            self.logger.info(f"UID {datapoint['UID']} category cls result '{depth3_val}' -> returning sentiment cls as '긍정' manually")
            sentiment_result = {
                "UID": datapoint["UID"],
                "task": "sentiment_classification",
                "result": {
                    "content": "긍정",
                    "raw": "소분류 만족 -> 감정분류 긍정",
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    },
                },
            }
        else:
            sentiment_result = self.sentiment_classification(datapoint)

        return [category_cls_result, sentiment_result]

    def _extract_digit(self, text) -> int:
        """Extract int number from string"""

        try:
            pattern = r"\d+"
            text = re.search(pattern, text)
            return text.group()
        except:
            # AttributeError
            return -1

    def _json_parse(self, text) -> dict:
        """Convert string into dict object"""

        if "```" in text:
            substrings = ["```", "json"]
            pattern = "|".join(map(re.escape, substrings))
            output = re.compile(pattern).sub("", text).strip()
        else:
            output = text

        # {} process in case of multiple {{}}
        def find_indices(s, char):
            return [i for i, c in enumerate(s) if c == char]

        start_idx = find_indices(output, "{")[-1]
        end_idx = find_indices(output, "}")[0]
        output2 = output[start_idx : end_idx + 1]

        output3 = eval(output2)
        return output3
