# Import necessary libraries
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
from translate import translate

# Define a FastAPI app
app = FastAPI()

# Define input data schema


class InputData(BaseModel):
    sourceLangCode_flores: str
    targetLangCode_flores: str
    textArray: List[str]

# Define the endpoint for the prediction


@app.post("/dt_translate_nllb")
async def dt_translate_nllb(data: InputData):
    # Return the prediction as a JSON response
    translations = await translate(data.sourceLangCode_flores, data.targetLangCode_flores, data.textArray)
    return {
        'status': 'success',
        'data': {'translationArray': translations}
    }

# curl -X POST -H "Content-Type: application/json" -d '{"sourceLangCode_flores": "eng_Latn", "targetLangCode_flores": "fra_Latn", "text": "This is a text: 1,2,3."}' http://localhost:8000/predict
