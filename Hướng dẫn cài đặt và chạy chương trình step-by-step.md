# SOS chatbot

Welcome to **Sos Req System**!

This website is designed to provide a rescue information.SOS Req system uses React for the front-end and NodeJS for the back-end and FastAPI for chatbot.

## Features:
- Upload file and store to chromaDB
- Create Retrieval QA with Agent langchain

## Requirements:
- Docker

Or 

- python >= 3.10
- langchain
- langchain-community
- gpt4all
- langchain_chroma
- langchain_text_splitters
- fastapi

- A modern web browser such as Google Chrome, Firefox, or Safari.
- A stable internet connection.
  
## How to use the website:
- Login into the system the valid account
  
## Installation
1. Install Docker and docker-compose `https://docs.docker.com/engine/install/`
2. Clone the repository: `git clone https://github.com/Tanh101/sos_chatbot`
3. Copy .env.exmaple to .env and add the api key
4. Install dependences
`pip install -r requirements.txt`
5. Start the development server: 
`docker run -d p 8001:8000 -t chatbot .`

OR using with 
`uvicorn main:app --reload --host $PORT`

## Support:
If you have any questions or run into any issues while using the website, please create new issue and describe it. We're always happy to help!

Thank you for choosing **SOS Req**. We hope you enjoy learning and growing as a developer with us!

# Clone source code
- git clone https://github.com/Tanh101/sos_chatbot
