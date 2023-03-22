# nllb-docker-rest
Use Ctranslate2 with NLLB-200 inside a docker container with GPUs

# Build container:
docker build -t nllb .

# Run container interactive mode:
docker run -it --rm --gpus 1,2,3,4 -p 8000:8000 -v $(pwd):/app nllb

# Test request:
curl -X POST -H "Content-Type: application/json" -d '{"sourceLangCode_flores": "eng_Latn", "targetLangCode_flores": "fra_Latn", "text": "This is a text: 1,2,3."}' http://localhost:8000/dt_translate_nllb
