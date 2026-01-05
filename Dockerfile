FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
LABEL Uday Jain <uday9k@gmail.com>
ENV TZ=Etc/UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common build-essential gcc wget
RUN pip install --upgrade pip setuptools wheel
WORKDIR /files
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt pip
RUN pip uninstall -y enum34
WORKDIR /files
EXPOSE 8878
RUN apt-get clean && rm -rf /var/lib/apt/lists/*
CMD ["jupyter", "notebook", "--port=8878", "--no-browser", "--ip=0.0.0.0", "--allow-root","--NotebookApp.token='whatever123'"]
