from utils import parse_answer, compare_answers

assert parse_answer("Answer: 1.0") == "1.0"
assert parse_answer("Answer: 1.9934343434") == "1.9934343434"
assert (
    parse_answer("index for the five year period ended 12/31/09 was -26.16%.")
    == "-26.16"
)
assert (
    parse_answer(
        "The percentage change in rental expenses from 2017 to 2018 was approximately 21.46%."
    )
    == "21.46"
)
assert (
    parse_answer(
        "```net sales were approximately 67.38% of the total consumer net sales in 2011.```"
    )
    == "67.38"
)
# assert parse_answer("```the 2003 gas transmission throughput would be approximately 644.37 bcf.```") == "644.37"


assert compare_answers(act="21.46%", exp="21%") is True
assert compare_answers(act="-6.37%", exp="-6.4%") is True
