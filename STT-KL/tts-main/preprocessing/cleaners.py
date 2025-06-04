import sys
import re
import abc
from typing import Union
from abc import abstractmethod

from .symbols import _alphabet, _punctuations, _not_end_punctuation, _numbers, _koreans
from .num2kor import NUM2KOR
from .conv_numbers import Numbers
from .eng2kor import enghanja_to_kor
from .pattern2kor import PATTERN2KOR
from .textpreprocessor import TextPreprocessor


class Cleaner(abc.ABC):

    @abstractmethod
    def __call__(self, text: Union[str, list]) -> Union[str, list]:
        """ Cleans text. """
        pass


class English(Cleaner):
    def __init__(self, alphabet=None):
        if not alphabet:
            self.accepted_chars = list(_alphabet) + list(_punctuations) + list(_numbers)
        self.numbers = Numbers(lang_ID='en',
                               comma='comma',
                               thousand='thousands')
        self.abbreviations = {
            'Mrs.': 'Mrs',
            'Mr.': 'Mr',
            'Dr.': 'Dr',
            'St.': 'St',
            'Co.': 'Co',
            'Jr.': 'Jr',
            'Maj.': 'Maj',
            'Gen.': 'Gen',
            'Drs.': 'Drs',
            'Rev.': 'Rev',
            'Lt.': 'Lt',
            'Hon.': 'Hon',
            'Sgt.': 'Sgt',
            'Capt.': 'Capt',
            'Esq.': 'Esq',
            'Ltd.': 'Ltd',
            'Col.': 'Col',
            'Ft.': 'Ft',
            'a.m.': 'a m',
            'p.m.': 'p m',
            'e.g.': 'e g',
            'i.e.': 'i e',
            ';': ',',
            ':': ','}

        self.abbreviations_pattern = self._get_abbreviation_pattern()

    def __call__(self, text: Union[str, list]) -> str:
        if isinstance(text, list):
            return [self._clean_line(t) for t in text]
        elif isinstance(text, str):
            return self._clean_line(text)
        else:
            raise TypeError(f'TextCleaner.clean() input must be list or str, not {type(text)}')

    def _get_abbreviation_pattern(self):
        return '|'.join(sorted(re.escape(k) for k in self.abbreviations))

    def _expand_abbreviations(self, text):
        return re.sub(self.abbreviations_pattern, lambda m: self.abbreviations.get(m.group(0)), text)

    def _filter_chars(self, text):
        return ''.join([c for c in text if c in self.accepted_chars])

    def _clean_line(self, text):
        text = self._filter_chars(text)
        text = self._expand_numbers(text)
        text = re.sub(self.abbreviations_pattern, lambda m: self.abbreviations.get(m.group(0)), text)
        if text.endswith(tuple(_not_end_punctuation)):
            text = text[:-1]
        return text + ' '

    def _expand_numbers(self, text):
        ends_with_dot = text.endswith('.')
        if ends_with_dot:
            text = text[:-1]
        text = self.numbers.expand_comma(text)
        text = self.numbers.expand_decimal_thousands(text)
        text = self.numbers.expand_decimal_hundreds(text)
        text = self.numbers.expand_decimal_point(text)
        text = self.numbers.expand_number(text)
        if ends_with_dot:
            text += '.'
        return text


class Korean(Cleaner):
    def __init__(self, alphabet=None, rest_at_the_front_of_sentence=False, rest_at_the_end_of_sentence=False, rest_characters_for_front_of_sentence=', ', rest_characters_for_end_of_sentence=', . '):
        self.tp = TextPreprocessor()
        self.n2k = NUM2KOR(self.tp)
        self.p2k = PATTERN2KOR(self.n2k, self.tp)
        self.rest_at_the_front_of_sentence = rest_at_the_front_of_sentence
        self.rest_at_the_end_of_sentence = rest_at_the_end_of_sentence
        self.rest_characters_for_front_of_sentence = rest_characters_for_front_of_sentence
        self.rest_characters_for_end_of_sentence = rest_characters_for_end_of_sentence
        # if not alphabet:
        #     self.accepted_chars = list(_koreans)

    def set_rest_at_the_end_of_sentence(self, rest_at_the_end_of_sentence):
        self.rest_at_the_end_of_sentence = rest_at_the_end_of_sentence

    def __call__(self, text: Union[str, list], train_or_play) -> Union[str, list]:
        if isinstance(text, list):
            return [self._clean_line(t, train_or_play).strip() for t in text]

        elif isinstance(text, str):
            return self._clean_line(text, train_or_play)

    # def _clean_line(self, text):
    def _clean_line(self, text, train_or_play):
        # if text.find('{{') != -1 and text.find('}}') != -1: # {{ }} is not work for cleaner.
        #     return text
        text = self.p2k.pattern_search_and_replace(self.tp.brackets_pattern, self.p2k.bracketspattern_to_korean, text)
        text = self.p2k.pattern_search_and_replace(self.tp.telephone_pattern, self.n2k.telephonnum_to_hanjanum, text)
        text = self.p2k.pattern_search_and_replace(self.tp.bankaccount_pattern, self.n2k.telephonnum_to_hanjanum, text)
        text = self.p2k.pattern_search_and_replace(self.tp.score_pattern, self.p2k.scorepattern_to_korean, text)
        text = self.p2k.pattern_search_and_replace(self.tp.A_to_B_pattern, self.p2k.fromtopattern_to_korean, text)
        text = self.p2k.pattern_search_and_replace(self.tp.date_pattern, self.p2k.datepattern_to_korean, text)
        text = self.p2k.pattern_search_and_replace(self.tp.time_pattern, self.p2k.timepattern_to_korean, text)
        text = self.p2k.pattern_search_and_replace(self.tp.math_pattern, self.p2k.mathtohan, text)
        if train_or_play == "play":
            print("Activate step read pattern.")
            text = self.p2k.pattern_search_and_replace(self.tp.step_read_pattern, self.p2k.step_words_add_comma, text) #only work for play

        #units2(written alphabet) convert
        for key, value in self.tp.units2_to_kor.items():
            pattern = re.compile(r'(\d+)' + key + '([^a-zA-Z])', re.IGNORECASE)
            text = re.sub(pattern, r'\1' + value + r'\2', text)

        text = self.p2k.pattern_search_and_replace(self.tp.english_number_pattern, self.n2k.initialnumber_to_englishnum, text)
        text = self.tp.convert_text(text)
        text = self.n2k.remove_num_comma(text)
        text = self.p2k.pattern_search_and_replace(self.tp.hangeulunit_pattern, self.n2k.hangeulunitnum_to_hangeulnum, text)
        text = self.p2k.pattern_search_and_replace(self.tp.incident_pattern, self.n2k.incidentdate_to_hangeul, text)
        text = self.n2k.sentence_num2kor(text)
        eng_pattern = re.compile(r'[a-z|A-Z]+')
        text = enghanja_to_kor(text, self.tp.english_tempdic, eng_pattern, self.tp.convert_full_translate_path(self.tp.textprocess_config['english_tempdic']))
        hanja_pattern = re.compile(r'[⺀-⺙⺛-⻳⼀-⿕々〇〡-〩〸-〺〻㐀-䶵一-鿃豈-鶴侮-頻並-龎]')
        text = enghanja_to_kor(text, self.tp.hanja_tempdic, hanja_pattern, self.tp.convert_full_translate_path(self.tp.textprocess_config['hanja_tempdic']))
        if self.rest_at_the_front_of_sentence:
            text = self.rest_characters_for_front_of_sentence + text
        if self.rest_at_the_end_of_sentence:
            text = text + self.rest_characters_for_end_of_sentence
        return text


class German(Cleaner):
    def __init__(self, alphabet=None):
        if not alphabet:
            self.accepted_chars = list(_alphabet) + list(_punctuations) + list(_numbers)
        self.numbers = Numbers(lang_ID='de',
                               comma='Komma',
                               thousand='tausend')
        self._date_re = re.compile(r'([0-9]{1,2}\.+)')
        self._time_re = re.compile(r'([0-9]{1,2}).([0-9]{1,2})(\s*Uhr)')

    def __call__(self, text: Union[str, list]) -> str:
        if isinstance(text, list):
            return [self._clean_line(t) for t in text]
        elif isinstance(text, str):
            return self._clean_line(text)
        else:
            raise TypeError(f'TextCleaner.clean() input must be list or str, not {type(text)}')

    def _filter_chars(self, text):
        return ''.join([c for c in text if c in self.accepted_chars])

    def _clean_line(self, text):
        text = self._filter_chars(text)
        text = self._expand_numbers(text)
        if text.endswith(tuple(_not_end_punctuation)):
            text = text[:-1]
        return text + ' '

    def _fix_time(self, m):
        if int(m.group(2)):
            return m.group(1) + m.group(3) + ' ' + m.group(2)  # 9 Uhr 30
        else:
            return m.group(1) + m.group(3)

    def _expand_date(self, m):
        num = int(m.group(0).replace('.', ''))
        if num < 20:
            return m.group(1).replace('.', 'ten')
        else:
            return m.group(1).replace('.', 'sten')

    def _expand_numbers(self, text):
        ends_with_dot = text.endswith('.')
        if ends_with_dot:
            text = text[:-1]
        text = self.numbers.expand_comma(text)
        text = re.sub(self._time_re, self._fix_time, text)
        text = self.numbers.expand_decimal_thousands(text)
        text = self.numbers.expand_decimal_hundreds(text)
        text = self.numbers.expand_decimal_point(text)
        text = re.sub(self._date_re, self._expand_date, text)
        text = self.numbers.expand_number(text)
        if ends_with_dot:
            text += '.'
        return text


# if __name__ == '__main__':
#     cleaner = Korean()

    # test_sentence = '안녕하세요, 코로나19 '
    # test_sentence = '최저임금위원회 정부 측 공익위원 중 간사를 맡은 권순원 숙명여대 교수(경영학부)는 14일 내년도 최저임금 인상률이 역대 최저인 1.5%로 결정된 데 대해 이같이 말했다. 최저임금 수준에 따라 같은 금액이 인상돼도 인상률은 천차만별일 수 있다는 뜻이다. 실제로 내년도 최저임금 인상액은 130원인데, 같은 금액이 인상된 1991년도에는 인상률이 18.8%에 달했다.'
    # test_sentence = '문재인정부 들어 최저임금 인상률은 극과 극을 달렸다. 2018∼2019년도 최저임금 심의에선 인상률이 두 자릿수를 기록했지만, 2020∼2021년도 심의에선 역대 최저 수준으로 떨어졌다. 정부 출범 당시 노동존중사회와 소득주도성장을 내세우며 2020년까지 ‘최저임금 1만원 시대’를 열겠다고 공약했지만, 현 정부 4년간 최저임금 연평균 인상률(7.7%)과 박근혜정부(2014∼2017년) 연평균 인상률(7.4%)은 크게 다르지 않다.'
    # test_sentence = '8차 전원회의로 본격적인 논의를 시작한 노사는 밤샘협상으로 날을 넘겨 9차 회의로 차수가 자동 변경됐다. 노사 인식 간극을 확인한 공익위원들은 심의촉진구간으로 ‘8620(+0.35%)∼9110원(+6.1%)’을 제시했고, 2014년 말 '
    # test_sentence = '11시에 도착합니다'
    # test_sentence = '추신수는 2014시즌을 앞두고 텍사스와 7년 1억3000만 달러의 대형 계약을 맺었다. 연봉 대비 활약에 대해선 논란이 따르기도 했지만, 추신수는 계속해서 자신의 자리를 지키며 제 몫을 다했다. 14시 6개, 14시간, 14, 15시, 15개, 15'
    # test_sentence = 'ㄱㄴㄷㄹ합니다'
    # test_sentence = '성내1동주민센터는 "22일(화) 16시에 ‘강동무료중식봉사회(회장: 정관훈)’ 주관 하에, 관내 어르신 60명을 대상으로 ‘독거 어르신께 달려가는 성내1동 짜장Day’ 사업을 시행한다."라고 밝혔다. 2,299,000, 90, 10살'
    # test_sentence = """이보배 곽민서 정수연 기자 = 지난달 소비자물가 상승률이 1.0%를 나타내며 6개월 만에 1%대로 올라섰다.
    #                 최장기간 장마 영향으로 농·축·수산물 가격이 2011년 3월 이후 9년 6개월 만에 가장 많이 올랐다.
    #                 6일 통계청 소비자물가 동향에 따르면 9월 소비자물가지수는 106.20(2015년=100)로 지난해 같은 달 대비 1.0% 상승했다.
    #                 이는 지난 3월(1.0%) 이후 최대 상승폭이다.
    #                 """

    #     test_sentence = """[아시아뉴스통신=장세희 기자] 강동구(구청장 이정훈) 성내1동주민센터는 "22일(화) 16시에 ‘강동무료중식봉사회(회장: 정관훈)’ 주관 하에 관내 어르신 60명을 대상으로 ‘독거 어르신께 달려가는 성내1동 짜장Day’ 사업을 시행한다."라고 밝혔다.
    # 이번 사업은 신종 코로나바이러스 감염증(코로나19)으로 인한 사회적 거리두기가 장기화됨에 따라, 지역 내 저소득 독거 어르신들의 우울증과 결식 가능성을 방지하기 위해 추진되었다. 특히, 위드코로나 시대에 맞춰 '비대면 배달서비스' 방식을 통해 어르신 식사 대접을 진행한다.
    # '강동무료중식봉사회' 회원들의 자발적 참여로 음식조리(짜장면 및 탕수육)와 비대면 배달서비스 등이 이루어진다. 또한, 어르신 건강에 필요한 지압기, 방석등도 전달되며 서비스 전후로 담당 복지플래너를 통해 어르신 건강 안부확인과 배달일정 안내 등이 함께 이루어질 예정이다.
    # 한원모 성내제1동장은 "이번 사업이 사회적 거리두기 장기화로 인해 마음 고생이 심했을 독거어르신들께 조금이나마 힘이 되길 바란다. 항상 안전수칙을 준수하고 감염병 홍보활동 등에 힘써 주민들이 마음 편히 복지서비스를 누릴 수 있도록 노력하겠다." 라고 말했다.
    # 한편, 이번 사업은 '성내1동 나눔가게-차이젠 해물왕짬뽕(대표 고원영)'의 후원을 통해 진행되던 독거어르신 식사대접 행사(이상 성내1동 짜장Day)의 연속적 성격을 띤다. 성내1동은 지난해부터 올해 초까지 월1회 정기행사를 통해 저소득 독거 어르신 총 240명을 대상으로 식사대접(짜장면, 볶음밥 등)과 건강상태 확인 등을 진행한 바 있다."""

    # test_sentence = 'this is test sentence'
    # test_sentence = '靑瓦臺 에서는 '
    # test_sentence = '김종인號 흔들린다’…①이슈 주춤 ②인물 부족 ③지지 하락'

    # test_sentence = '16시, 16개, 16송이'
    # test_sentence = 'ㅁㅊㄴㅂㅈㄷㅊㅇㅇ ㅡㅏ '
    # test_sentence = '30송이 20송이 52개 1시 4시 1개 10송이 1송이 10팀 10개교 11개12개 1:1 12:54 ① 6개월 2,299,000, 90, 10살 14시 6개, 14시간, 14, 15시, 15개, 15 0.35%'

    # test_sentence = "원주율 π는 대략 3.14159이다"
    # print('Result')
    # result = cleaner(test_sentence, "play")
    # print(result)
