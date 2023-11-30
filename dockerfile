FROM python:3.10.10

ADD . /code

WORKDIR /code
 
RUN pip install -r requirements.txt -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

CMD ["sh","-c","python3","main.py"]