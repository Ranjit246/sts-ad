# Pull tensorflow image with Python3
FROM pytorch/pytorch:2.2.1-cuda11.8-cudnn8-devel

# Set the working directory to /app
WORKDIR /app

# Transfer content from current dir to /app in container
ADD . /app

# Install audio libraries
RUN apt-get update && apt-get install -y python3.11 libsndfile1 libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0 ffmpeg git

# Install PyTorch
RUN pip install torch torchvision torchaudio nltk gunicorn 

RUN python setup_nltk.py

# Clone Git repositories
RUN git clone https://github.com/AI4Bharat/IndicTrans2.git
RUN git clone https://github.com/VarunGumma/IndicTransTokenizer.git

# Install the IndicTransTokenizer package in editable mode
RUN pip3 install --editable /app/IndicTransTokenizer

# Install python packages
RUN pip install -r requirements.txt

# Start Gunicorn server using custom configuration file
CMD ["gunicorn", "-c", "gunicorn_config.py", "app:app"]