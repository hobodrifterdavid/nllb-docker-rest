# nllb-docker-rest
Use Ctranslate2 with NLLB-200 inside a docker container with GPUs. You will need a gpu-enabled version of Docker installed, see NVIDIA docs for instructions for setting that up. 

### Prepare NLLB model
Download the model to nllb-200-3.3B folder.

Convert it using the tool from ctranslate2 lib:
ct2-transformers-converter --model nllb-200-3.3B/ --output_dir nllb-200-3.3B-converted

### Build container:
docker build -t nllb .

### Run container interactive mode with GPUs 0 and 1:
docker run -it --rm --gpus '"device=0,1"' -p 8000:8000 -v $(pwd):/app nllb

If running with different (number) of gpus, also adjust device_index in translate.py

### Test request:
chmod +x test.sh
./test.sh
