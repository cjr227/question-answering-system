FROM python:3.11-slim

WORKDIR /src

ADD requirements.txt /src/

RUN apt-get update && apt-get install -y libpq-dev gcc

RUN pip install -r requirements.txt

COPY ./ ./

EXPOSE 8050

CMD ["gunicorn", "-w 2", "-b :8050", "src.app:server", "-t 300"]
