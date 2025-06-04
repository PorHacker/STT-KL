import re
import json
import pandas as pd
import ruamel.yaml

class TextPreprocessor:
    '''
    The TextPreprocessor:
     - convert units to kor
     - convert + or - sign
     - convert English Upper letters
    '''

    def load_json(self, dict_path):
        with open(dict_path, encoding='utf-8') as json_file:
            dict_data = json.load(json_file)
        return dict_data

    def convert_full_rule_path(self, sub_path):
        rule_dir_recognition_pattern = re.compile(r'\$rule_dir')
        return re.sub(rule_dir_recognition_pattern, self.rule_dir, sub_path)

    def convert_full_translate_path(self, sub_path):
        translate_dir_recognition_pattern = re.compile(r'\$translate_dir')
        return self.convert_full_rule_path(re.sub(translate_dir_recognition_pattern, self.translate_dir, sub_path))

    def convert_full_pattern_path(self, sub_path):
        pattern_dir_recognition_pattern = re.compile(r'\$pattern_dir')
        return self.convert_full_rule_path(re.sub(pattern_dir_recognition_pattern, self.pattern_dir, sub_path))

    def convert_full_number_path(self, sub_path):
        number_dir_recognition_pattern = re.compile(r'\$number_dir')
        return self.convert_full_rule_path(re.sub(number_dir_recognition_pattern, self.number_dir, sub_path))

    def convert_full_dictionary_path(self, sub_path):
        dictionary_dir_recognition_pattern = re.compile(r'\$dictionary_dir')
        return self.convert_full_rule_path(re.sub(dictionary_dir_recognition_pattern, self.dictionary_dir, sub_path))

    def __init__(self):
        yaml = ruamel.yaml.YAML()
        with open(str('./preprocessing/text/config/textprocess_config.yaml'), 'rb') as data_yaml:
            self.textprocess_config = yaml.load(data_yaml)
        self.rule_dir = self.textprocess_config['rule_dir']
        self.translate_dir = self.textprocess_config['translate_dir']
        self.pattern_dir = self.textprocess_config['pattern_dir']
        self.number_dir = self.textprocess_config['number_dir']
        self.dictionary_dir = self.textprocess_config['dictionary_dir']

        #translate
        self.currency_to_kor = self.load_json(self.convert_full_translate_path(self.textprocess_config['currency_to_kor']))
        self.units_to_kor = self.load_json(self.convert_full_translate_path(self.textprocess_config['units_to_kor']))
        self.units2_to_kor = self.load_json(self.convert_full_translate_path(self.textprocess_config['units2_to_kor']))
        self.mathsymbol_to_kor = self.load_json(self.convert_full_translate_path(self.textprocess_config['mathsymbol_to_kor']))
        # self.eng_upper_to_kor = eng_upper_to_kor
        # self.hangeul_measure_num = self.load_json(self.convert_full_rule_path(self.textprocess_config['hangeul_measure_num']))
        self.kor_processing = self.load_json(self.convert_full_translate_path(self.textprocess_config['kor_processing']))
        self.char_translate_korean = self.load_json(self.convert_full_translate_path(self.textprocess_config['char_translate_korean']))
        self.engword_to_kor = self.load_json(self.convert_full_translate_path(self.textprocess_config['engword_to_kor']))
        self.hanja1800_to_kor = self.load_json(self.convert_full_translate_path(self.textprocess_config['hanja1800_to_kor']))
        self.heading = self.load_json(self.convert_full_translate_path(self.textprocess_config['heading']))
        self.string_test = self.load_json(self.convert_full_translate_path(self.textprocess_config['string_test']))
        self.english_tempdic = self.load_json(self.convert_full_translate_path(self.textprocess_config['english_tempdic']))
        self.hanja_tempdic = self.load_json(self.convert_full_translate_path(self.textprocess_config['hanja_tempdic']))
        self.unique_word_dic = self.load_json(self.convert_full_translate_path(self.textprocess_config['unique_word_dic']))
        self.hiragana_to_kor = self.load_json(self.convert_full_translate_path(self.textprocess_config['hiragana_to_kor']))
        self.katagana_to_kor = self.load_json(self.convert_full_translate_path(self.textprocess_config['katagana_to_kor']))

        #pattern
        self.bankaccount_pattern = pd.read_csv(self.convert_full_pattern_path(self.textprocess_config['bankaccount_pattern']), encoding='UTF-8', header=None)
        self.date_pattern = pd.read_csv(self.convert_full_pattern_path(self.textprocess_config['date_pattern']),
                                        encoding='UTF-8', header=None)
        self.A_to_B_pattern = pd.read_csv(self.convert_full_pattern_path(self.textprocess_config['A_to_B_pattern']),
                                          encoding='UTF-8', header=None)
        self.score_pattern = pd.read_csv(self.convert_full_pattern_path(self.textprocess_config['score_pattern']),
                                         encoding='UTF-8', header=None)
        self.telephone_pattern = pd.read_csv(self.convert_full_pattern_path(self.textprocess_config['telephone_pattern']), encoding='UTF-8', header=None)
        self.time_pattern = pd.read_csv(self.convert_full_pattern_path(self.textprocess_config['time_pattern']), encoding='UTF-8', header=None)
        self.hangeulunit_pattern = pd.read_csv(self.convert_full_pattern_path(self.textprocess_config['hangeulunit_pattern']),
                                               encoding='UTF-8', header=None)
        self.incident_pattern = pd.read_csv(self.convert_full_pattern_path(self.textprocess_config['incident_pattern']),
                                           encoding='UTF-8', header=None)
        self.math_pattern = pd.read_csv(self.convert_full_pattern_path(self.textprocess_config['math_pattern']),
                                            encoding='UTF-8', header=None)
        self.brackets_pattern = pd.read_csv(self.convert_full_pattern_path(self.textprocess_config['brackets_pattern']),
                                        encoding='UTF-8', header=None)
        self.english_number_pattern = pd.read_csv(self.convert_full_pattern_path(self.textprocess_config['english_number_pattern']),
                                            encoding='UTF-8', header=None)
        self.step_read_pattern = pd.read_csv(self.convert_full_pattern_path(self.textprocess_config['step_read_pattern']),
            encoding='UTF-8', header=None)

        #dictionary
        self.hangeul_number_measure = pd.read_csv(self.convert_full_dictionary_path(self.textprocess_config['hangeul_number_measure']),
                                           encoding='UTF-8', header=None)
        self.hanja_number_measure = pd.read_csv(self.convert_full_dictionary_path(self.textprocess_config['hanja_number_measure']), encoding='UTF-8', header=None)
        self.incident_clue_words = pd.read_csv(
            self.convert_full_dictionary_path(self.textprocess_config['incident_clue_words']), encoding='UTF-8', header=None)
        self.score_related_words = pd.read_csv(
            self.convert_full_dictionary_path(self.textprocess_config['score_related_words']), encoding='UTF-8',
            header=None)
        self.step_read_words = pd.read_csv(
            self.convert_full_dictionary_path(self.textprocess_config['step_read_words']), encoding='UTF-8',
            header=None)
        self.hangulnum_dan_words = pd.read_csv(
            self.convert_full_dictionary_path(self.textprocess_config['hangulnum_dan_words']), encoding='UTF-8',
            header=None)

        #korean number
        # self.hangeul_kisu1 = self.load_json(self.convert_full_number_path(self.textprocess_config['hangeul_kisu1']))
        self.hangeul_kisu2 = self.load_json(self.convert_full_number_path(self.textprocess_config['hangeul_kisu2']))
        self.hanja_kisu = self.load_json(self.convert_full_number_path(self.textprocess_config['hanja_kisu']))
        self.english_kisu = self.load_json(self.convert_full_number_path(self.textprocess_config['english_kisu']))
        self.designnumber_to_number = self.load_json(self.convert_full_number_path(self.textprocess_config['designnumber_to_number']))
        # self.hanjanumber2 = self.load_json(self.convert_full_number_path(self.textprocess_config['hanjanumber2']))

    def mapping_from_dict(self, text, dict_):
        for key, value in dict_.items():
            text = text.replace(key, value)
        return text

    # def mapping_from_dict2(self, text, dict_):
    #     for key, value in dict_.items():
    #         # indexNo = text.find(key)
    #         # if indexNo > 0:
    #         text = text.replace(key, value)
    #     return text

    def convert_text(self, text):
        self.text = text
        self.text = self.mapping_from_dict(self.text, self.unique_word_dic)
        self.text = self.mapping_from_dict(self.text, self.kor_processing)
        self.text = self.mapping_from_dict(self.text, self.currency_to_kor)
        self.text = self.mapping_from_dict(self.text, self.units_to_kor)    # force translate
        self.text = self.mapping_from_dict(self.text, self.mathsymbol_to_kor)
        self.text = self.mapping_from_dict(self.text, self.designnumber_to_number)
        self.text = self.mapping_from_dict(self.text, self.engword_to_kor)
        self.text = self.mapping_from_dict(self.text, self.hanja1800_to_kor)
        self.text = self.mapping_from_dict(self.text, self.heading)
        self.text = self.mapping_from_dict(self.text, self.string_test)
        self.text = self.mapping_from_dict(self.text, self.hiragana_to_kor)
        self.text = self.mapping_from_dict(self.text, self.katagana_to_kor)
        self.text = self.mapping_from_dict(self.text, self.char_translate_korean)

        return self.text
