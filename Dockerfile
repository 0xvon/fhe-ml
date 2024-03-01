FROM python:3.9.12

RUN apt-get update && apt-get install -y \ 
    cmake \
    libfreetype6-dev \
    libpng-dev \
    libopenblas-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install notebook

# COPY ./src /app/
# CMD ["python", "./main.py"]
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
