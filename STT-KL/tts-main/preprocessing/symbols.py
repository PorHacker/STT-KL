## This is very sensitive data. Please consult to admin before editing.
_vowels = 'iyɨʉɯuɪʏʊeøɘəɵɤoɛœɜɞʌɔæɐaɶɑɒᵻ'
_non_pulmonic_consonants = 'ʘɓǀɗǃʄǂɠǁʛ'
_pulmonic_consonants = 'pbtdʈɖcɟkɡqɢʔɴŋɲɳnɱmʙrʀⱱɾɽɸβfvθðszʃʒʂʐçʝxɣχʁħʕhɦɬɮʋɹɻjɰlɭʎʟ'
_suprasegmentals = 'ˈˌːˑ'
_other_symbols = 'ʍwɥʜʢʡɕʑɺɧ'
_diacrilics = 'ɚ˞ɫ'
_phonemes = sorted(list(
    _vowels + _non_pulmonic_consonants + _pulmonic_consonants + _suprasegmentals + _other_symbols + _diacrilics))
# _punctuations = '!,-.:;? ' # for wavernn
_punctuations = '!,-.:;? \'()' # for hifigan
_alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzäüößÄÖÜ'
all_phonemes = sorted(list(_phonemes) + list(_punctuations))
_not_end_punctuation = ',-.:; '
_numbers = '1234567890'
_hangul = ''.join([chr(i) for i in range(0xac00, 0xd7a4)])
_koreans = 'ㄱㄲㄳㄴㄵㄶㄷㄸㄹㄺㄻㄼㄾㅀㅁㅂㅃㅄㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ'
# _korean_punctuations = '!?., '
_korean_phonemes = ['aa', 'c0', 'cc', 'ch', 'ee', 'h0', 'ii', 'k0', 'kf', 'kh', 'kk', 'ks', 'lb', 'lh', 'lk', 'll', 'lm', 'lp',
         'ls', 'lt', 'mf', 'mm', 'nc', 'nf', 'nh', 'nn', 'ng', 'oh', 'oo', 'p0', 'pf', 'ph', 'pp', 'ps', 'qq', 'rr', 's0',
         'ss', 't0', 'tf', 'th', 'tt', 'uu', 'vv', 'wa', 'we', 'wi', 'wo', 'wq', 'wv', 'xi', 'xx', 'ya', 'ye', 'yo',
         'yq', 'yu', 'yv']
ONS = ['k0', 'kk', 'nn', 't0', 'tt', 'rr', 'mm', 'p0', 'pp',
       's0', 'ss', 'oh', 'c0', 'cc', 'ch', 'kh', 'th', 'ph', 'h0']
NUC = ['aa', 'qq', 'ya', 'yq', 'vv', 'ee', 'yv', 'ye', 'oo', 'wa',
       'wq', 'wo', 'yo', 'uu', 'wv', 'we', 'wi', 'yu', 'xx', 'xi', 'ii']
COD = ['', 'kf', 'kk', 'ks', 'nf', 'nc', 'nh', 'tf',
       'll', 'lk', 'lm', 'lb', 'ls', 'lt', 'lp', 'lh',
       'mf', 'pf', 'ps', 's0', 'ss', 'oh', 'c0', 'ch',
       'kh', 'th', 'ph', 'h0']
NUCv = ['aa*', 'qq*', 'ya*', 'yq*', 'vv*', 'ee*', 'yv*', 'ye*', 'oo*', 'wa*',
       'wq*', 'wo*', 'yo*', 'uu*', 'wv*', 'we*', 'wi*', 'yu*', 'xx*', 'xi*', 'ii*']
pitch = ['1', '2', '3']
# voice_act = ['[[', ']]']
voice_act = ['[[', ']]', '{{', '}}']
NUC2 = ['i1', 'i2', 'i3', 'i4', 'i5', 'e1', 'e2', 'e3', 'e4', 'e5', 'q1', 'q2', 'q3', 'q4', 'q5', 'a1', 'a2', 'a3', 'a4', 'a5', 'x1', 'x2', 'x3', 'x4', 'x5', 'v1', 'v2', 'v3', 'v4', 'v5',  'u1', 'u2', 'u3', 'u4', 'u5', 'o1', 'o2', 'o3', 'o4', 'o5']
NUC3 = ['iⅠ', 'iⅡ', 'iⅢ', 'iⅣ', 'iⅤ', 'eⅠ', 'eⅡ', 'eⅢ', 'eⅣ', 'eⅤ', 'qⅠ', 'qⅡ', 'qⅢ', 'qⅣ', 'qⅤ', 'aⅠ', 'aⅡ', 'aⅢ', 'aⅣ', 'aⅤ', 'xⅠ', 'xⅡ', 'xⅢ', 'xⅣ', 'xⅤ', 'vⅠ', 'vⅡ', 'vⅢ', 'vⅣ', 'vⅤ', 'uⅠ', 'uⅡ', 'uⅢ', 'uⅣ', 'uⅤ', 'oⅠ', 'oⅡ', 'oⅢ', 'oⅣ', 'oⅤ']
# NUC2 = ['ii1', 'ii2', 'ii3', 'ii4', 'ii5', 'ee1', 'ee2', 'ee3', 'ee4', 'ee5', 'qq1', 'qq2', 'qq3', 'qq4', 'qq5', 'aa1', 'aa2', 'aa3', 'aa4', 'aa5', 'xx1', 'xx2', 'xx3', 'xx4', 'xx5', 'vv1', 'vv2', 'vv3', 'vv4', 'vv5',  'uu1', 'uu2', 'uu3', 'uu4', 'uu5', 'oo1', 'oo2', 'oo3', 'oo4', 'oo5',
#         'ye1', 'ye2', 'ye3', 'ye4', 'ye5', 'yq1', 'yq2', 'yq3', 'yq4', 'yq5', 'ya1', 'ya2', 'ya3', 'ya4', 'ya5', 'yv1', 'yv2', 'yv3', 'yv4', 'yv5', 'yu1', 'yu2', 'yu3', 'yu4', 'yu5', 'yo1', 'yo2', 'yo3', 'yo4', 'yo5',
#         'wi1', 'wi2', 'wi3', 'wi4', 'wi5', 'wo1', 'wo2', 'wo3', 'wo4', 'wo5', 'wq1', 'wq2', 'wq3', 'wq4', 'wq5', 'we1', 'we2', 'we3', 'we4', 'we5', 'wa1', 'wa2', 'wa3', 'wa4', 'wa5', 'wv1', 'wv2', 'wv3', 'wv4', 'wv5', 'xi1', 'xi2', 'xi3', 'xi4', 'xi5']
# NUC3 = ['iiⅠ', 'iiⅡ', 'iiⅢ', 'iiⅣ', 'iiⅤ', 'eeⅠ', 'eeⅡ', 'eeⅢ', 'eeⅣ', 'eeⅤ', 'qqⅠ', 'qqⅡ', 'qqⅢ', 'qqⅣ', 'qqⅤ', 'aaⅠ', 'aaⅡ', 'aaⅢ', 'aaⅣ', 'aaⅤ', 'xxⅠ', 'xxⅡ', 'xxⅢ', 'xxⅣ', 'xxⅤ', 'vvⅠ', 'vvⅡ', 'vvⅢ', 'vvⅣ', 'vvⅤ', 'uuⅠ', 'uuⅡ', 'uuⅢ', 'uuⅣ', 'uuⅤ', 'ooⅠ', 'ooⅡ', 'ooⅢ', 'ooⅣ', 'ooⅤ',
#         'yeⅠ', 'yeⅡ', 'yeⅢ', 'yeⅣ', 'yeⅤ', 'yqⅠ', 'yqⅡ', 'yqⅢ', 'yqⅣ', 'yqⅤ', 'yaⅠ', 'yaⅡ', 'yaⅢ', 'yaⅣ', 'yaⅤ', 'yvⅠ', 'yvⅡ', 'yvⅢ', 'yvⅣ', 'yvⅤ', 'yuⅠ', 'yuⅡ', 'yuⅢ', 'yuⅣ', 'yuⅤ', 'yoⅠ', 'yoⅡ', 'yoⅢ', 'yoⅣ', 'yoⅤ',
#         'wiⅠ', 'wiⅡ', 'wiⅢ', 'wiⅣ', 'wiⅤ', 'woⅠ', 'woⅡ', 'woⅢ', 'woⅣ', 'woⅤ', 'wqⅠ', 'wqⅡ', 'wqⅢ', 'wqⅣ', 'wqⅤ', 'weⅠ', 'weⅡ', 'weⅢ', 'weⅣ', 'weⅤ', 'waⅠ', 'waⅡ', 'waⅢ', 'waⅣ', 'waⅤ', 'wvⅠ', 'wvⅡ', 'wvⅢ', 'wvⅣ', 'wvⅤ', 'xiⅠ', 'xiⅡ', 'xiⅢ', 'xiⅣ', 'xiⅤ']
BREATH_IN = ['@', '$', '&']