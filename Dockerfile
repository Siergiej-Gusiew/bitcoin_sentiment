FROM python:3.8.5
ARG PROJNAME=bitcoin_sentiment
ENV PROJNAME=${PROJNAME}
RUN mkdir /${PROJNAME}
WORKDIR /${PROJNAME}

# install deployment dependencies
COPY requirements.txt .
RUN pip install --no-dependencies --no-cache-dir -r requirements.txt

# copy all the files to the container
COPY . .
# tell the port number the container should expose
EXPOSE 5000

# run the command
CMD ["python", "./app.py"]
