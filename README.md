## intro

	- a piece of code working with a dataset (data/train.json) from https://github.com/czyssrs/ConvFinQA
	- it aism to answer questions invluded in 
	
### reports

	- html and csv files are created for every run
	- html includes each document processed, actual and expected answer, errors as well as accuracy summary

### run
	- `poetry install`
	- `mv sample.env .env`
	- update .env manually # see below
		- (easy) when using openai, add OPENAI_API_KEY to .env
		- when using Bedrock, setup aws cli and add credentials profile
	- `python main.py`

### providers
	- openai
	- bedrock
	
### models
	- gpt-4
	- amazon.titan-text-express-v1
	- or another from from respective provider

### agents

	- simple
	- agent_keeping_history
	- agent_developer

### running with local model

	- there is an option to run with a local model but it is super slow
	- it does not generate a report either
	- `run_with_local_model.py`
