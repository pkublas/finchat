from report import parse_answer


assert parse_answer("Answer: 1.0") == "1.0"
assert parse_answer("Answer: 1.9934343434") == "1.9934343434"
