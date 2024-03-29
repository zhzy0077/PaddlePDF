FROM paddlepaddle/paddle:2.6.0

RUN pip install "paddleocr>=2.0.1"
RUN pip install "filetype"

RUN pip install "watchdog"

COPY ./paddlepdf.py /paddlepdf.py

ENTRYPOINT ["/usr/bin/python", "/paddlepdf.py"]