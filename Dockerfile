#docker 制作文件
FROM python:3.6-stretch

WORKDIR /app

RUN pip install tensorflow tensorlayer -i https://pypi.tuna.tsinghua.edu.cn/simple/

ENV EXAM_NO='011151605' EXAMINNE_NO='18110101153447' USERNAME=filehelper TF_CPP_MIN_LOG_LEVEL=3 LANG=C.UTF-8

COPY model.npz main.py main1.py model.py ./

CMD python main.py
