import re
from .num2kor import NUM2KOR
from .textpreprocessor import TextPreprocessor

class PATTERN2KOR:
  def __init__(self, num2kor, textpreprocessor):
    self.n2k = num2kor
    self.tp = textpreprocessor
    # self.date_pattern = date_pattern
    self.pattern = re.compile('')

  def num2kor(self, num):
    return self.n2k.sentence_num2kor(num.group())

  # def si2kor(self, si):
  #   return self.tp.mapping_from_dict(si.group(), self.tp.hangeul_measure_num)

  def datepattern_to_korean(self, data):
    try:
      data_to_han = re.sub(self.pattern, r'\1년 \2월 \3일', data.group())
      data_to_han = re.sub(r'(^\d{2})년|\D+(\d{2})년', r'이천\1년', data_to_han)
    except:
      data_to_han = re.sub(self.pattern, r'\1월 \2일', data.group())
    # data_to_han = re.sub(r'\d+', self.num2kor, data_to_han)
    return data_to_han

  def fromtopattern_to_korean(self, data):
    middle_char = re.sub(self.pattern, r'\3', data.group())
    # backward_chars = re.sub(self.pattern, r'\5', data.group())
    if middle_char == '~' or middle_char == '∼':  # ascii or unicode
      data_to_han = re.sub(self.pattern, r'\2 에서 \4', data.group())
    elif middle_char == '-':
      data_to_han = re.sub(self.pattern, r'\2 \4', data.group())
    else:
      data_to_han = re.sub(self.pattern, r'\1', data.group())
    return data_to_han

  def bracketspattern_to_korean(self, data):
    datagroup = data.group()
    data_to_han = re.sub(self.pattern, r'\2', data.group())
    return data_to_han


  def scorepattern_to_korean(self, data):
    datastring = data.string
    middle_char = re.sub(self.pattern, r'\3', data.group())
    # forward_chars = re.sub(self.pattern, r'\2', data.group())
    backward_chars = re.sub(self.pattern, r'\5', data.group())
    if middle_char == ':' or middle_char == '-':
      clue_str = re.findall(r'[ 가-힣]+', data.string)
      for index, row in self.tp.score_related_words.iterrows():
        unit = row.values[0]
        if unit in backward_chars:
          if '에' in backward_chars:
            data_to_han = re.sub(self.pattern, r'\1', data.group())
            break
          else:
            data_to_han = re.sub(self.pattern, r'\2 대 \4 \5', data.group())
            break
      else:
        data_to_han = re.sub(self.pattern, r'\1', data.group())
    else:
      data_to_han = re.sub(self.pattern, r'\1', data.group())
    return data_to_han

  def mathtohan(self, data):
    front_char = re.sub(self.pattern, r'\1', data.group())
    # for key, value in self.tp.mathsymbol_to_kor.items():
      # text = text.replace(key, value)
    if front_char == '-':
      data_to_han = re.sub(self.pattern, r'마이너스\2', data.group())
        # return data_to_han
    else:
      data_to_han = re.sub(self.pattern, r'\1\2', data.group())
    return data_to_han

  def step_words_add_comma(self, data):
    find = False
    # datastring = data.string
    forward_chars = re.sub(self.pattern, r'\2', data.group())
    # if middle_char == ':' or middle_char == '-':
    # clue_str = re.findall(r'[ 가-힣]+', data.string)
    for index, row in self.tp.step_read_words.iterrows():
      unit = row.values[0]
      l_unit = len(unit)
      if unit in forward_chars[-l_unit:]:
      #   if '에' in backward_chars:
      #     data_to_han = re.sub(self.pattern, r'\1', data.group())
      #     break
      #   else:
        data_to_han = re.sub(self.pattern, r'\2, ', data.group())
        find=True
        break
    # else:
    #   data_to_han = re.sub(self.pattern, r'\1', data.group())
    if not find:
      data_to_han = re.sub(self.pattern, r'\1', data.group())
    return data_to_han

  def timepattern_to_korean(self, data):
    try:
      data_to_han = re.sub(self.pattern, r' \1시 \2분 \3초', data.group())
    except:
      data_to_han = re.sub(self.pattern, r' \1시 \2분', data.group())
    # data_to_han = re.sub(r'0\d', self.n2k.telephonnum_to_hanjanum, data_to_han)
    # data_to_han = re.sub(r'( \d+시)', self.si2kor, data_to_han)
    # data_to_han = re.sub(r'\d+', self.num2kor, data_to_han)
    return data_to_han

  # def teletohan(self, data):
  #   pattern = re.compile(r'(010[-]\d{4}[-]\d{4})')
  #   data_to_han = re.sub(r'(010)(\d{4})(\d{4})', r' \1-\2-\3', data.group())
  #   repattern = pattern.search(data_to_han)
  #   return self.n2k.numtohan2(repattern)

  def pattern_search_and_replace(self, pattern, func, datas):
    transform_data=datas
    for index, row in pattern.iterrows():
      self.pattern=re.compile(row[0])
      transform_data=re.sub(self.pattern, func, transform_data)
    return transform_data

if __name__ == '__main__':
  dates = """
  1921.03.05
  2021. 03. 05
  2021.03.05.
  2021. 03. 05.
  2020.2.4
  2020. 2. 4
  2020.2.4.
  2020. 2. 4.
  09.01.12
  물19.07.28
  19. 07. 28
  19.06.28.
  19. 06. 28.
  18.1.9
  18. 1. 9
  18.1.9.
  18. 1. 9.
  1911-02-12
  1910-3-8
  09-10-28
  08-4-6
  """
  times = """
  12:58
  12:58:32
  05:02
  05:12:21
  23:12
  23:12:11
  2:03
  2:33:05

  """
  tele = """
  010-4567-1234
  010.4567.1234
  010 4567 1234
  01045671234
  """
  fromto = """
  1~2
  """
  tp = TextPreprocessor()
  n2k = NUM2KOR(tp)
  a = PATTERN2KOR(n2k, tp)
  tele_to_han = a.pattern_search_and_replace(tp.telephone_pattern, a.n2k.telephonnum_to_hanjanum, tele)
  print(tele_to_han)
  date_to_han=a.pattern_search_and_replace(tp.date_pattern, a.datepattern_to_korean, dates)
  print(date_to_han)
  time_to_han = a.pattern_search_and_replace(tp.time_pattern, a.timepattern_to_korean, times)
  print(time_to_han)
  fromto_han = a.pattern_search_and_replace(tp.A_to_B_pattern, a.fromtopattern_to_korean, fromto)
  print(fromto_han)