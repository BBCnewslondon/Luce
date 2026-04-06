FROM python:3.12-slim

WORKDIR /app

COPY requirements.bot.txt /app/requirements.bot.txt
RUN pip install --no-cache-dir -r /app/requirements.bot.txt

COPY . /app

CMD ["python", "-m", "execution.mtf_oanda_bot"]
