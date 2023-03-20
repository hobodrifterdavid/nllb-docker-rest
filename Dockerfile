# docker build -t nllb .
# docker run -it --rm --gpus 1 -p 8000:8000 -v $(pwd):/app nllb
# docker run -it --rm -p 8000:8000 -v $(pwd):/app nllb

FROM nvidia/cuda:11.2.2-runtime-ubuntu20.04

# Install necessary dependencies
RUN apt-get update && \
    apt-get install -y python3-pip

# Set the working directory
WORKDIR /app

# Copy the app code and requirements file
COPY . /app
# COPY requirements.txt .

# Install the app dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Expose the FastAPI port
EXPOSE 8000

# Start the FastAPI app using Uvicorn web server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]


# uvicorn app:app --host 0.0.0.0 --port 8000