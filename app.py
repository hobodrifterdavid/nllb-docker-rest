# Import necessary libraries
from typing import List, Tuple
from fastapi import FastAPI
from pydantic import BaseModel
from translate import translate_batch_async
import asyncio

# Define a FastAPI app
app = FastAPI()

# Define input data schema


class InputData(BaseModel):
    sourceLangCode_flores: str
    targetLangCode_flores: str
    textArray: List[str]


active_translation_requests: List[Tuple[str, str, List[str], asyncio.Future]] = [
]  # a list to store translation requests


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
    active_translation_requests.append(
        (src_lang_flores, tgt_lang_flores, textArr, future))
    result = await future
    return result


async def batch_translate():
    global active_translation_requests

    # allow the event loop to handle some network requests?
    await asyncio.sleep(0.1)

    while True:
        if active_translation_requests:
            requestsProcessingNow = len(active_translation_requests)
            results = await translate_batch_async(
                [src for src, _, _, _ in active_translation_requests],
                [tgt for _, tgt, _, _ in active_translation_requests],
                [text for _, _, text, _ in active_translation_requests]
            )

            for i, (_, _, _, future) in enumerate(active_translation_requests[:requestsProcessingNow]):
                future.set_result(results[i])
            # remove the processed tasks:
            active_translation_requests = active_translation_requests[requestsProcessingNow:]
        else:
            await asyncio.sleep(0.1)


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(batch_translate())
