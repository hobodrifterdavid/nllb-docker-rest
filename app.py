# Import necessary libraries
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
from translate import translate_sync, translate_async, translate_batch_async

import asyncio
from typing import List

# Define a FastAPI app
app = FastAPI()

# Define input data schema


class InputData(BaseModel):
    sourceLangCode_flores: str
    targetLangCode_flores: str
    textArray: List[str]


translations = []  # a list to store translation requests


@app.post("/dt_translate_nllb")
async def dt_translate_nllb(data: InputData):
    # Return the prediction as a JSON response
    task = asyncio.create_task(process_translation(
        data.sourceLangCode_flores, data.targetLangCode_flores, data.textArray))
    await task

    translationArray = task.result()

    return {
        'status': 'success',
        'data': {'translationArray': translationArray}
    }


async def process_translation(src_lang_flores: str, tgt_lang_flores: str, textArr: List[str]) -> List[str]:
    future = asyncio.get_running_loop().create_future()
    translations.append((src_lang_flores, tgt_lang_flores, textArr, future))
    result = await future
    return result


async def batch_translate():
    global translations

    # allow the event loop to handle some network requests?
    await asyncio.sleep(0.1)

    while True:
        if translations:
            processingNow = len(translations)
            batch_src = [src for src, _, _, _ in translations]
            batch_tgt = [tgt for _, tgt, _, _ in translations]
            batch_text = [text for _, _, text, _ in translations]
            results = await translate_batch_async(batch_src, batch_tgt, batch_text)

            for i, (_, _, _, future) in enumerate(translations[:processingNow]):
                future.set_result(results[i])
            # remove the processed tasks:
            translations = translations[processingNow:]
        else:
            await asyncio.sleep(0.1)


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(batch_translate())
