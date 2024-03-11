## intro

	- an app that is working with a dataset (data/train.json) obtained from https://github.com/czyssrs/ConvFinQA
	- it aims to answer questions included in train.json file
	- and generates html and csv file for every run in `reports` directory
	- html report includes details of each document processed, actual and expected answer, errors as well as accuracy summary
	- an app can work with OpenAI and AWS Bedrock services by making appropriate config changed in `.env` file, as described below

### setup

	- duplicate sample.env and create `.env` file
	- update `.env`
		- when using OpenAI, just add `OPENAI_API_KEY`
		- when using Bedrock, setup aws cli locally add aws profile name to `AWS_BEDROCK_PROFILE_NAME`
		- update add other parameters as per their description below	
		
##### .env

```
	OPENAI_API_KEY="sk-123"
	AWS_BEDROCK_PROFILE_NAME=default
	MODEL_PROVIDER=choose "openai" OR "bedrock"
	MODEL_NAME=choose between "amazon.titan-text-express-v1" or "gpt-4"
	AGENT=choose between "simple", "agent_keeping_history" or "agent_developer", these agents are described in html files under reports dir
	NO_OF_DOCUMENTS_TO_PROCESS=number of documents to process from train.json file
	VERBOSE=False, whether to show output from running commands
```

### run

#### Docker (recommended)

	- It has been tested on Windows only. On MacOs/Linux try included Makefile.
	- When using Bedrock, you need to have local aws config and credentials files setup

Build an image:
	- `docker build -t finchat:0.1 -f Dockerfile .`

Run a container
	`docker run --env-file .env --name finchat_app --volume ${PWD}/data:/app/data --volume ${PWD}/reports:/app/reports --volume ${HOME}/.aws:/root/.aws --rm finchat:0.1

#### manually

	- `poetry install`
	- create `.env` file as per the setup
	- `python main.py`

#### running with local model

	- there is an option to run with a local model but it is super slow
	- it does not generate a report either
	- `run_with_local_model.py`
