FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Warsaw

RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip install pyautogui

RUN apt-get update && apt-get install -y \
    wget unzip curl gnupg2 libnss3 libgconf-2-4 libxss1 fonts-liberation \
    libasound2 libatk1.0-0 libatk-bridge2.0-0 libgtk-3-0 libx11-xcb1 python3 python3-pip \
    --no-install-recommends && rm -rf /var/lib/apt/lists/*

# Instalacja Google Chrome
RUN wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | apt-key add - && \
    echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google-chrome.list && \
    apt-get update && apt-get install -y google-chrome-stable

RUN apt-get update && apt-get install -y git
RUN apt-get update && apt-get install -y python3-tk python3-dev
RUN apt-get update && apt-get install -y libxss1 libappindicator3-1 libindicator7

# Instalacja ChromeDriver
RUN CHROME_DRIVER_VERSION=$(curl -sS chromedriver.storage.googleapis.com/LATEST_RELEASE) && \
    wget -O /tmp/chromedriver_linux64.zip https://chromedriver.storage.googleapis.com/${CHROME_DRIVER_VERSION}/chromedriver_linux64.zip && \
    unzip /tmp/chromedriver_linux64.zip -d /usr/local/bin/ && \
    rm /tmp/chromedriver_linux64.zip && \
    chmod +x /usr/local/bin/chromedriver

# Ustawienie katalogu aplikacji
WORKDIR /app

# Kopiowanie plików projektu
COPY *.py /app/
COPY part_2.json /app/
COPY requirements.txt /app/
# Instalacja pakietów Pythonowych
RUN pip3 install --no-cache-dir -r requirements.txt

# Uruchomienie aplikacji
CMD ["python3", "6_Ludzik.py"]
