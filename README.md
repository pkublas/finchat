## intro

	- a piece of code working with a dataset (data/train.json) from https://github.com/czyssrs/ConvFinQA
	
### reports

	- html and csv fiels are created for every run
	- html includes each document processed, actual and expected answer, errors as well as accuracy summary

### run
	- `poetry install`
	- `mv sample.env .env`
	- update .env manually
	- when using Bedrock, setup aws cli
	- `python main.py`
	
### agents

	- "simple"
	- "agent_keeping_history"
	- "agent_developer"

### running with local model

	- there is an option to run with a local model but ti is super slow
	- also, it does not generate a report
	- `run_with_local_model.py`
