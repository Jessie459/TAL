import json
import random

import regex as re


def fix_json(input_str):
    # Add double quotes around keys using regex
    fixed_str = re.sub(r"(\w+):", r'"\1":', input_str)

    # Add double quotes around string values if necessary and wrap int/float values in []
    def format_value(match):
        key, value, comma = match.groups()
        value = value.strip()
        # Check if value is an integer or float
        if re.match(r"^-?\d+(\.\d+)?$", value):
            value = f"[{value}]"
        # Check if value is a boolean or null
        elif re.match(r"^(true|false|null)$", value, re.IGNORECASE):
            pass  # leave as is
        else:
            # Add quotes around string values
            value = f'"{value}"'
        return f"{key}: {value}{comma}"

    fixed_str = re.sub(r'(".*?"):(.*?)(,|})', format_value, fixed_str)

    return fixed_str


def verify(s, target_sequence):
    # Count the occurrences of the target sequence
    count = s.count(target_sequence)

    # Check if the target sequence appears exactly twice
    return count == 2


def is_int_between_0_and_10(s):
    try:
        num = int(s)
        return 0 <= num <= 10
    except ValueError:
        return False


def mllm_output_to_dict(input_string, give_up_parsing=False):
    """
    Args:
        input_string (str): actually the output of the mllm model to be parsed
        output_file_name (str): The name of the output file.
    """
    # Catch for gpt4v rate_limit_exceeded error
    if input_string == "rate_limit_exceeded":
        return "rate_limit_exceeded"

    # Define the delimiters
    delimiter = "||V^=^V||"

    if input_string.count(delimiter) == 2:
        if not verify(input_string, delimiter):
            print("The required delimiters were not found correctly in the string.")
            return False
        # Extract the content between the delimiters
        start_index = input_string.find(delimiter) + len(delimiter)
        end_index = input_string.rfind(delimiter)
    else:
        # find the json mannually
        # some mllm tends not to output the delimiters, but it does output the json contents
        # so we will find the json content mannually
        start_index = input_string.find("{")
        end_index = input_string.rfind("}") + 1
        if start_index == -1 or end_index == 0:
            # json not found
            # some mllm tends to output only a list of scores like [6, 0],
            # this time we will just get the scores and ignore the reasoning (other part of the json)
            start_index = input_string.find("[")
            end_index = input_string.rfind("]") + 1
            if give_up_parsing:  # if we want to give up parsing
                guessed_value = random.randint(0, 10)
                print(f"Failed to find the json content in the string. Guess a value : {guessed_value}.")
                json_content = {"score": [guessed_value], "reasoning": f"guess_if_cannot_parse | {input_string}"}
                json_str = json.dumps(json_content)
                input_string = json_str
                start_index = 0
                end_index = len(json_str)
            elif re.match(r"^\[\d+, ?\d+\]$", input_string[start_index:end_index]):
                scores = json.loads(input_string[start_index:end_index])
                if not isinstance(scores, list):
                    scores = [scores]
                json_content = {"score": scores, "reasoning": "System: output is simply a list of scores"}
                json_str = json.dumps(json_content)
                input_string = json_str
                start_index = 0
                end_index = len(json_str)
            elif is_int_between_0_and_10(input_string):  # if output is simply a number
                scores = [int(input_string)]
                json_content = {"score": scores, "reasoning": "System: output is simply a number"}
                json_str = json.dumps(json_content)
                input_string = json_str
                start_index = 0
                end_index = len(json_str)
            else:
                print("Failed to find the json content in the string.")
                return False

    # Check if we found two delimiters
    if start_index != -1 and end_index != -1 and start_index != end_index:
        # Extract the JSON string
        json_str = input_string[start_index:end_index].strip()
        json_str = json_str.replace("\n", "")
        # Parse the JSON string into a dictionary
        try:
            new_data = json.loads(json_str)
            if not isinstance(new_data["score"], list):
                new_data["score"] = [new_data["score"]]
        except:
            print("Now fixing: ", json_str)
            try:
                new_data = json.loads(fix_json(json_str))
                return new_data
            except:
                print("Error: Cannot fix", json_str)
                return False
        return new_data
    else:
        print("The required delimiters were not found correctly in the string.")
        return False
