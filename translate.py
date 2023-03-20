import ctranslate2
import transformers
from typing import Dict, List, Union

translator = ctranslate2.Translator(
    "nllb-200-3.3B-converted", device='auto', compute_type="float16", device_index=[0, 1, 2, 3])

# Cache the tokenizer, in case it's expensive to create them..
tokenizers: Dict[str, Union[transformers.PreTrainedTokenizer,
                 transformers.PreTrainedTokenizerFast]] = {}


async def translate(src_lang_flores: str, tgt_lang_flores: str, textArr: List[str]) -> List[str]:

    if (src_lang_flores not in tokenizers):
        tokenizers[src_lang_flores] = transformers.AutoTokenizer.from_pretrained(
            "nllb-200-3.3B", src_lang=src_lang_flores)

    tokenizer = tokenizers[src_lang_flores]

    target_prefix = [tgt_lang_flores]

    results = translator.translate_batch(
        [tokenizer.convert_ids_to_tokens(
            tokenizer.encode(text)) for text in textArr],
        target_prefix=[target_prefix]*len(textArr),
        max_batch_size=512
        )
    
    # 1. (self: ctranslate2._ext.Translator, source: List[List[str]], target_prefix: Optional[List[Optional[List[str]]]] = None, *, max_batch_size: int = 0, batch_type: str = 'examples', asynchronous: bool = False, beam_size: int = 2, patience: float = 1, num_hypotheses: int = 1, length_penalty: float = 1, coverage_penalty: float = 0, repetition_penalty: float = 1, no_repeat_ngram_size: int = 0, disable_unk: bool = False, suppress_sequences: Optional[List[List[str]]] = None, end_token: Optional[str] = None, prefix_bias_beta: float = 0, max_input_length: int = 1024, max_decoding_length: int = 256, min_decoding_length: int = 1, use_vmap: bool = False, return_scores: bool = False, return_attention: bool = False, return_alternatives: bool = False, min_alternative_expansion_prob: float = 0, sampling_topk: int = 1, sampling_temperature: float = 1, replace_unknowns: bool = False) -> Union[List[ctranslate2._ext.TranslationResult], List[ctranslate2._ext.AsyncTranslationResult]]

    targets = [result.hypotheses[0][1:] for result in results]

    translations = [tokenizer.decode(
        tokenizer.convert_tokens_to_ids(target)) for target in targets]
    print(translations)
    return translations


# googleToFlores200Codes = {
#     'af': 'afr_Latn',
#     'sq': 'als_Latn',
#     'am': 'amh_Ethi',
#     'ar': 'arb_Arab',
#     'hy': 'hye_Armn',
#     'az': 'azj_Latn', # 'North Azerbaijani'
#     'eu': 'eus_Latn',
#     'be': 'bel_Cyrl',
#     'bn': 'ben_Beng',
#     'bs': 'bos_Latn',
#     'bg': 'bul_Cyrl',
#     'ca': 'cat_Latn',
#     'ceb': 'ceb_Latn',
#     'zh-CN': 'zho_Hans',
#     'zh-TW': 'zho_Hant',
#     # Nope?
#     # co: 'Corsican',
#     'hr': 'hrv_Latn',
#     'cs': 'ces_Latn',
#     'da': 'dan_Latn',
#     'nl': 'nld_Latn',
#     'en': 'eng_Latn',
#     'eo': 'epo_Latn',
#     'et': 'est_Latn',
#     'fi': 'fin_Latn',
#     'fr': 'fra_Latn',
#     # Nope
#     # fy: 'Frisian',
#     'gl': 'glg_Latn',
#     'ka': 'kat_Geor',
#     'de': 'deu_Latn',
#     'el': 'ell_Grek',
#     'gu': 'guj_Gujr',
#     'ht': 'hat_Latn',
#     'ha': 'hau_Latn',
#     # Nope
#     # haw: 'Hawaiian',
#     'iw': 'heb_Hebr',
#     'hi': 'hin_Deva',
#     # Nope:
#     # hmn: 'Hmong',
#     'hu': 'hun_Latn',
#     'is': 'isl_Latn',
#     'ig': 'ibo_Latn',
#     'id': 'ind_Latn',
#     'ga': 'gle_Latn',
#     'it': 'ita_Latn',
#     'ja': 'jpn_Jpan',
#     'jv': 'jav_Latn',
#     'kn': 'kan_Knda',
#     'kk': 'kaz_Cyrl',
#     'km': 'khm_Khmr',
#     'ko': 'kor_Hang',
#     # Kurmanji - Northern Kurdish. There's also Sorani (central Kurdish)
#     'ku': 'kmr_Latn',
#     'ky': 'kir_Cyrl',
#     'lo': 'lao_Laoo',
#     # Nope:
#     # la: 'Latin',
#     'lv': 'lvs_Latn',
#     'lt': 'lit_Latn',
#     'lb': 'ltz_Latn',
#     'mk': 'mkd_Cyrl',
#     'mg': 'plt_Latn',
#     'ms': 'zsm_Latn',
#     'ml': 'mal_Mlym',
#     'mt': 'mlt_Latn',
#     'mi': 'mri_Latn',
#     'mr': 'mar_Deva',
#     'mn': 'khk_Cyrl',
#     'my': 'mya_Mymr',
#     'ne': 'npi_Deva',
#     # Bokmal, not newnorsk:
#     'no': 'nob_Latn',
#     'ny': 'nya_Latn',
#     # Southern Pashto
#     'ps': 'pbt_Arab',
#     'fa': 'pes_Arab',
#     'pl': 'pol_Latn',
#     'pt': 'por_Latn',
#     # Nope:
#     # pa: 'Punjabi',
#     'ro': 'ron_Latn',
#     'ru': 'rus_Cyrl',
#     'sm': 'smo_Latn',
#     'gd': 'gla_Latn',
#     # Note: cyrillic!
#     'sr': 'srp_Cyrl',
#     # Nope:
#     # st: 'Sesotho',
#     'sn': 'sna_Latn',
#     'sd': 'snd_Arab',
#     'si': 'sin_Sinh',
#     'sk': 'slk_Latn',
#     'sl': 'slv_Latn',
#     'so': 'som_Latn',
#     'es': 'spa_Latn',
#     'su': 'sun_Latn',
#     'sw': 'swh_Latn',
#     'sv': 'swe_Latn',
#     'tl': 'Ttgl_Latn',
#     'tg': 'tgk_Cyrl',
#     'ta': 'tam_Taml',
#     'te': 'tel_Telu',
#     'th': 'tha_Thai',
#     'tr': 'tur_Latn',
#     'uk': 'ukr_Cyrl',
#     'ur': 'urd_Arab',
#     'uz': 'uzn_Latn',
#     'vi': 'vie_Latn',
#     'cy': 'cym_Latn',
#     'xh': 'xho_Latn',
#     # 'Eastern' Yiddish
#     'yi': 'ydd_Hebr',
#     'yo': 'yor_Latn',
#     'zu': 'zul_Latn',
#     # New:
#     'rw': 'kin_Latn',
#     'or': 'ory_Orya',
#     # Tatarstan, there is also Crimean Tatar:
#     'tt': 'tat_Cyrl',
#     'tk': 'tuk_Latn',
#     'ug': 'uig_Arab',
# }
