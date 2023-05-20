import asyncio
import ctranslate2
import transformers
from typing import Dict, List, Union
from stopes_snippet.split_clean import splitAndClean


translator = ctranslate2.Translator(
    "nllb-200-3.3B-converted", device='auto', compute_type="float16", device_index=[0, 1])

# Cache the tokenizer, in case it's expensive to create them..
tokenizers: Dict[str, Union[transformers.PreTrainedTokenizer,
                 transformers.PreTrainedTokenizerFast]] = {}


async def translate_batch_async(src_lang_flores: List[str], tgt_lang_flores: List[str], batches_textArrArr: List[List[str]]) -> List[str]:

    # Flaten the passed batches down to simple flat arrays
    flatBatches_textArr: List[str] = []
    # flores lang codes..
    flatBatches_src_lang_flores: List[str] = []
    # flores lang codes..
    flatBatches_tgt_lang_flores: List[str] = []

    for i, batch_textArr in enumerate(batches_textArrArr):
        # two-letter Google translate codes..
        thisSrcLang_flores = src_lang_flores[i]
        thisDestLang_flores = tgt_lang_flores[i]

        for text in batch_textArr:
            flatBatches_textArr.append(text)
            flatBatches_src_lang_flores.append(src_lang_flores[i])
            flatBatches_tgt_lang_flores.append(tgt_lang_flores[i])

    # Further divide each string to sentences using a sentence splitter..

    sentences: List[str] = []
    # How many sentences were in each passed string..
    sentences_counts: List[int] = []
    sentences_src_lang_flores: List[str] = []
    sentences_tgt_lang_flores: List[str] = []

    for i, thisTranslateText in enumerate(flatBatches_textArr):

        thisSrcLang_flores = flatBatches_src_lang_flores[i]
        thisTgtLang_flores = flatBatches_tgt_lang_flores[i]
        theseSentences: List[str] = splitAndClean(
            thisSrcLang_flores, thisTranslateText)

        sentences_counts.append(len(theseSentences))
        sentences.extend(theseSentences)
        sentences_src_lang_flores.extend(
            [thisSrcLang_flores]*len(theseSentences))
        sentences_tgt_lang_flores.extend(
            [thisTgtLang_flores]*len(theseSentences))

    # Tokenize the sentences

    sentences_tokensied: List[List[str]] = []

    for i, sentence in enumerate(sentences):

        thisSrcLang_flores = sentences_src_lang_flores[i]

        if (thisSrcLang_flores not in tokenizers):
            tokenizers[thisSrcLang_flores] = transformers.AutoTokenizer.from_pretrained(
                "nllb-200-3.3B", src_lang=thisSrcLang_flores)

        tokenizer = tokenizers[thisSrcLang_flores]

        thisSentenceTokens = tokenizer.convert_ids_to_tokens(
            tokenizer.encode(sentence))
        sentences_tokensied.extend([thisSentenceTokens])

    # ok, let's translate already..

    print(f"Processing batch: {len(sentences_tokensied)}")

    def sync_func():
        return translator.translate_batch(
            sentences_tokensied,
            target_prefix=[[thisDestLang_flores]
                           for thisDestLang_flores in sentences_tgt_lang_flores],
            max_batch_size=128
        )

    # Run sync_func asyncronously, so we don't block the event loop.
    # Allows other requests to be handled meanwhile.
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(None, lambda: sync_func())

    targets = [result.hypotheses[0][1:] for result in results]

    sentences_translations = [tokenizer.decode(
        tokenizer.convert_tokens_to_ids(target)) for target in targets]

    # Let's reconstruct back to where we split with the sentence splitter..

    flatBatches_translations: List[str] = []

    for count in sentences_counts:
        # Joining with a space, ideally this would be language specific..
        flatBatches_translations.append(
            ' '.join(sentences_translations[:count]))
        # Remove these items from the list
        sentences_translations[:count] = []

    # Let's assemble back into the passed batches..
    batches_translationsArrArr: List[List[str]] = []

    # Loop over the input batches
    for batch_textArr in batches_textArrArr:
        batch_translationArr: List[str] = []
        # loop over the strings passed in each batch
        for text in batch_textArr:
            batch_translationArr.append(flatBatches_translations.pop(0))
        batches_translationsArrArr.append(batch_translationArr)

    return batches_translationsArrArr

# def translate_sync(src_lang_flores: str, tgt_lang_flores: str, textArr: List[str]) -> List[str]:

#     if (src_lang_flores not in tokenizers):
#         tokenizers[src_lang_flores] = transformers.AutoTokenizer.from_pretrained(
#             "nllb-200-3.3B", src_lang=src_lang_flores)

#     tokenizer = tokenizers[src_lang_flores]

#     target_prefix = [tgt_lang_flores]

#     arg1 = [tokenizer.convert_ids_to_tokens(
#             tokenizer.encode(text)) for text in textArr]
#     arg2 = target_prefix = [target_prefix]*len(textArr)

#     print(f"Processing batch: {len(arg2)}")

#     results = translator.translate_batch(
#         arg1,
#         target_prefix=arg2,
#         max_batch_size=128
#     )

#     targets = [result.hypotheses[0][1:] for result in results]

#     translations = [tokenizer.decode(
#         tokenizer.convert_tokens_to_ids(target)) for target in targets]

#     return translations


# async def translate_async(src_lang_flores: str, tgt_lang_flores: str, textArr: List[str]) -> List[str]:

#     if (src_lang_flores not in tokenizers):
#         tokenizers[src_lang_flores] = transformers.AutoTokenizer.from_pretrained(
#             "nllb-200-3.3B", src_lang=src_lang_flores)

#     tokenizer = tokenizers[src_lang_flores]

#     target_prefix = [tgt_lang_flores]

#     arg1 = [tokenizer.convert_ids_to_tokens(
#             tokenizer.encode(text)) for text in textArr]
#     arg2 = target_prefix = [target_prefix]*len(textArr)

#     print(f"Processing batch: {len(arg2)}")

#     def sync_func():
#         return translator.translate_batch(
#             arg1,
#             target_prefix=arg2,
#             max_batch_size=128
#         )

#     # Run sync_func asyncronously, so we don't block the event loop.
#     # Allows other requests to be handled meanwhile.
#     loop = asyncio.get_event_loop()
#     results = await loop.run_in_executor(None, lambda: sync_func())

#     targets = [result.hypotheses[0][1:] for result in results]

#     translations = [tokenizer.decode(
#         tokenizer.convert_tokens_to_ids(target)) for target in targets]
#     # print(translations)
#     return translations

googleToFlores200Codes = {
    'af': 'afr_Latn',
    'sq': 'als_Latn',
    'am': 'amh_Ethi',
    'ar': 'arb_Arab',
    'hy': 'hye_Armn',
    'az': 'azj_Latn',  # 'North Azerbaijani'
    'eu': 'eus_Latn',
    'be': 'bel_Cyrl',
    'bn': 'ben_Beng',
    'bs': 'bos_Latn',
    'bg': 'bul_Cyrl',
    'ca': 'cat_Latn',
    'ceb': 'ceb_Latn',
    'zh-CN': 'zho_Hans',
    'zh-TW': 'zho_Hant',
    # Nope?
    # co: 'Corsican',
    'hr': 'hrv_Latn',
    'cs': 'ces_Latn',
    'da': 'dan_Latn',
    'nl': 'nld_Latn',
    'en': 'eng_Latn',
    'eo': 'epo_Latn',
    'et': 'est_Latn',
    'fi': 'fin_Latn',
    'fr': 'fra_Latn',
    # Nope
    # fy: 'Frisian',
    'gl': 'glg_Latn',
    'ka': 'kat_Geor',
    'de': 'deu_Latn',
    'el': 'ell_Grek',
    'gu': 'guj_Gujr',
    'ht': 'hat_Latn',
    'ha': 'hau_Latn',
    # Nope
    # haw: 'Hawaiian',
    'iw': 'heb_Hebr',
    'hi': 'hin_Deva',
    # Nope:
    # hmn: 'Hmong',
    'hu': 'hun_Latn',
    'is': 'isl_Latn',
    'ig': 'ibo_Latn',
    'id': 'ind_Latn',
    'ga': 'gle_Latn',
    'it': 'ita_Latn',
    'ja': 'jpn_Jpan',
    'jv': 'jav_Latn',
    'kn': 'kan_Knda',
    'kk': 'kaz_Cyrl',
    'km': 'khm_Khmr',
    'ko': 'kor_Hang',
    # Kurmanji - Northern Kurdish. There's also Sorani (central Kurdish)
    'ku': 'kmr_Latn',
    'ky': 'kir_Cyrl',
    'lo': 'lao_Laoo',
    # Nope:
    # la: 'Latin',
    'lv': 'lvs_Latn',
    'lt': 'lit_Latn',
    'lb': 'ltz_Latn',
    'mk': 'mkd_Cyrl',
    'mg': 'plt_Latn',
    'ms': 'zsm_Latn',
    'ml': 'mal_Mlym',
    'mt': 'mlt_Latn',
    'mi': 'mri_Latn',
    'mr': 'mar_Deva',
    'mn': 'khk_Cyrl',
    'my': 'mya_Mymr',
    'ne': 'npi_Deva',
    # Bokmal, not newnorsk:
    'no': 'nob_Latn',
    'ny': 'nya_Latn',
    # Southern Pashto
    'ps': 'pbt_Arab',
    'fa': 'pes_Arab',
    'pl': 'pol_Latn',
    'pt': 'por_Latn',
    # Nope:
    # pa: 'Punjabi',
    'ro': 'ron_Latn',
    'ru': 'rus_Cyrl',
    'sm': 'smo_Latn',
    'gd': 'gla_Latn',
    # Note: cyrillic!
    'sr': 'srp_Cyrl',
    # Nope:
    # st: 'Sesotho',
    'sn': 'sna_Latn',
    'sd': 'snd_Arab',
    'si': 'sin_Sinh',
    'sk': 'slk_Latn',
    'sl': 'slv_Latn',
    'so': 'som_Latn',
    'es': 'spa_Latn',
    'su': 'sun_Latn',
    'sw': 'swh_Latn',
    'sv': 'swe_Latn',
    'tl': 'Ttgl_Latn',
    'tg': 'tgk_Cyrl',
    'ta': 'tam_Taml',
    'te': 'tel_Telu',
    'th': 'tha_Thai',
    'tr': 'tur_Latn',
    'uk': 'ukr_Cyrl',
    'ur': 'urd_Arab',
    'uz': 'uzn_Latn',
    'vi': 'vie_Latn',
    'cy': 'cym_Latn',
    'xh': 'xho_Latn',
    # 'Eastern' Yiddish
    'yi': 'ydd_Hebr',
    'yo': 'yor_Latn',
    'zu': 'zul_Latn',
    # New:
    'rw': 'kin_Latn',
    'or': 'ory_Orya',
    # Tatarstan, there is also Crimean Tatar:
    'tt': 'tat_Cyrl',
    'tk': 'tuk_Latn',
    'ug': 'uig_Arab',
}
