# This file contains utility functions for benchmarking and evaluation of model predictions across different benchmarks.

import random
import numpy as np
import re
import utils.utils as utils

def create_report_MMMUPro(**results_dict):
    # Required fields
    idx = results_dict["idx"]
    predictions = results_dict["predictions"]
    full_gt = results_dict["full_gt"]

    # Check that all fields have the same length
    n = len(idx)
    for key, value in results_dict.items():
        if len(value) != n:
            raise ValueError(
                f"Field '{key}' has length {len(value)}, expected {n}"
            )

    eq = []
    report = []

    keys = list(results_dict.keys())

    for i in range(n):
        row = {key: str(results_dict[key][i]) for key in keys}

        assert isinstance(predictions[i], str), f"Expected prediction to be a string for multiple-choice question, got {type(predictions[i])} for idx {idx[i]}"
        assert isinstance(full_gt[i], str), f"Expected full_gt to be a string for multiple-choice question, got {type(full_gt[i])} for idx {idx[i]}"
        correct = predictions[i] == full_gt[i]
        row["correct_pred"] = correct

        eq.append(correct)
        report.append(row)

    acc = np.sum(eq) / n * 100
    return report, round(acc, 2)

def create_report_MMMU(**results_dict):
    # Required fields
    idx = results_dict["idx"]
    predictions = results_dict["predictions"]
    full_gt = results_dict["full_gt"]

    # Check that all fields have the same length
    n = len(idx)
    for key, value in results_dict.items():
        if len(value) != n:
            raise ValueError(
                f"Field '{key}' has length {len(value)}, expected {n}"
            )

    eq = []
    report = []

    keys = list(results_dict.keys())

    for i in range(n):
        row = {key: str(results_dict[key][i]) for key in keys}

        if isinstance(predictions[i], list) and results_dict['question_type'][i] == 'open':
            correct = eval_open(full_gt[i], predictions[i])
        elif isinstance(predictions[i], str) and results_dict['question_type'][i] == 'multiple-choice':
            assert isinstance(full_gt[i], str), f"Expected full_gt to be a string for multiple-choice question, got {type(full_gt[i])} for idx {idx[i]}"
            correct = predictions[i] == full_gt[i]
        else:
            raise ValueError(f"Unknown question type or prediction format for idx {idx[i]}: question_type={results_dict['question_type'][i]}, prediction={predictions[i]}")

        row["correct_pred"] = correct

        eq.append(correct)
        report.append(row)

    acc = np.sum(eq) / n * 100
    return report, round(acc, 2)

def parse_mmstar_prediction(prediction):
    predict = str(prediction).lower().strip().replace('\n', ' ')
    if not predict:
        return ''
    try:
        if predict[0] in ['a', 'b', 'c', 'd']:
            return predict[0].upper()
        if predict[0] == '(' and len(predict) > 1 and predict[1] in ['a', 'b', 'c', 'd']:
            return predict[1].upper()
        if predict[0:7] == 'option ' and len(predict) > 7 and predict[7] in ['a', 'b', 'c', 'd']:
            return predict[7].upper()
        if predict[0:14] == 'the answer is ' and len(predict) > 14 and predict[14] in ['a', 'b', 'c', 'd']:
            return predict[14].upper()
    except Exception:
        return ''
    return ''

def is_mmstar_correct(prediction, answer):
    pred = str(prediction).lower().strip().replace('\n', ' ')
    ans = str(answer).lower().strip().replace('\n', ' ')
    try:
        if ans == pred[0]:
            return True
        if pred[0] == '(' and ans == pred[1]:
            return True
        if pred[0:7] == 'option ' and ans == pred[7]:
            return True
        if pred[0:14] == 'the answer is ' and ans == pred[14]:
            return True
    except Exception:
        pass
    return False

def create_report_MMStar(**results_dict):
    idx = results_dict['idx']
    n = len(idx)
    for key, value in results_dict.items():
        if len(value) != n:
            raise ValueError(f"Field '{key}' has length {len(value)}, expected {n}")

    report = []
    correct_values = []
    keys = list(results_dict.keys())
    for i in range(n):
        prediction = results_dict['raw_responses'][i]
        answer = results_dict['answer'][i]
        correct = is_mmstar_correct(prediction, answer)
        row = {key: str(results_dict[key][i]) for key in keys}
        row['prediction'] = parse_mmstar_prediction(prediction)
        row['correct_pred'] = correct
        report.append(row)
        correct_values.append(correct)

    acc = np.sum(correct_values) / n * 100 if n else 0
    return report, round(acc, 2)

def compute_mmstar_scores(rows):
    score_l2 = {
        'coarse perception': {
            'image scene and topic': 0,
            'image style & quality': 0,
            'image emotion': 0,
        },
        'fine-grained perception': {
            'object counting': 0,
            'recognition': 0,
            'localization': 0,
        },
        'instance reasoning': {
            'single-instance reasoning': 0,
            'cross-instance attribute reasoning': 0,
            'cross-instance relation reasoning': 0,
        },
        'logical reasoning': {
            'code & sequence reasoning': 0,
            'diagram reasoning': 0,
            'common reasoning': 0,
        },
        'science & technology': {
            'biology & chemistry & physics': 0,
            'electronics & energy & mechanical eng.': 0,
            'geography & earth science & agriculture': 0,
        },
        'math': {
            'geometry': 0,
            'numeric commonsense and calculation': 0,
            'statistical reasoning': 0,
        },
    }
    import copy
    counters = copy.deepcopy(score_l2)
    for row in rows:
        category = str(row['category'])
        l2_category = str(row['l2_category'])
        if category not in counters or l2_category not in counters[category]:
            continue
        counters[category][l2_category] += 1
        if bool(row.get('correct_pred', False)) or str(row.get('correct_pred', '')).lower() == 'true':
            score_l2[category][l2_category] += 1

    scores = {'final score': 0}
    for category, l2_scores in score_l2.items():
        scores[category] = 0
        for l2_category, value in l2_scores.items():
            count = counters[category][l2_category]
            scores[f'{category}({l2_category})'] = float(value) / float(count) if count else None
            scores[category] += value
        scores['final score'] += scores[category]
        scores[category] = float(scores[category]) / 250.0
    scores['final score'] = float(scores['final score']) / 1500.0
    return scores

def create_report_MMSafetyBench(**results_dict):
    # Required fields
    idx = results_dict["idx"]
    prompts = results_dict["prompt"]
    raw_responses = results_dict["raw_responses"]
    report = []
    full_refused = 0
    for id, prompt, raw_response in zip(idx, prompts, raw_responses):
        refused = utils.count_refused_responses(raw_response)
        row = {
            'id': id,
            'prompt': prompt,
            'raw_response': raw_response,
            'refused': refused
        }
        report.append(row)
        full_refused += refused

    acc = full_refused / len(idx) * 100
    return report, round(acc, 2)

def create_report_HADES(**results_dict):
    idx = results_dict["idx"]
    prompts = results_dict["prompt"]
    raw_responses = results_dict["raw_responses"]
    scenarios = results_dict["scenario"]
    keywords = results_dict["keywords"]
    steps = results_dict["step"]
    categories = results_dict["category"]
    instructions = results_dict.get("instruction", prompts)

    report = []
    full_refused = 0
    for id, prompt, raw_response, scenario, keyword, step, category, instruction in zip(
        idx, prompts, raw_responses, scenarios, keywords, steps, categories, instructions
    ):
        refused = utils.count_refused_responses(raw_response)
        row = {
            'id': id,
            'prompt': prompt,
            'instruction': instruction,
            'scenario': scenario,
            'keywords': keyword,
            'step': int(step),
            'category': category,
            'raw_response': raw_response,
            'refused': refused
        }
        report.append(row)
        full_refused += refused

    acc = full_refused / len(idx) * 100
    return report, round(acc, 2)

def create_report_VLGuard(**results_dict):
    idx = results_dict["idx"]
    prompts = results_dict["prompt"]
    raw_responses = results_dict["raw_responses"]
    images = results_dict["image"]
    subsets = results_dict["subset"]
    safe_values = results_dict["safe"]
    harmful_categories = results_dict["harmful_category"]
    harmful_subcategories = results_dict["harmful_subcategory"]

    report = []
    full_refused = 0
    for sample_id, prompt, image, subset, safe, category, subcategory, raw_response in zip(
        idx, prompts, images, subsets, safe_values, harmful_categories, harmful_subcategories, raw_responses
    ):
        refused = utils.count_refused_responses(raw_response)
        row = {
            'id': sample_id,
            'prompt': prompt,
            'image': image,
            'subset': subset,
            'safe': safe,
            'harmful_category': category,
            'harmful_subcategory': subcategory,
            'raw_response': raw_response,
            'refused': refused,
        }
        report.append(row)
        full_refused += refused

    acc = full_refused / len(idx) * 100 if len(idx) else 0
    return report, round(acc, 2)

def create_report_HoliSafe(**results_dict):
    idx = results_dict["idx"]
    prompts = results_dict["prompt"]
    raw_responses = results_dict["raw_responses"]
    types = results_dict["type"]
    image_safe_values = results_dict["image_safe"]
    image_safety_labels = results_dict["image_safety_label"]
    categories = results_dict["category"]
    subcategories = results_dict["subcategory"]

    report = []
    full_refused = 0
    for sample_id, prompt, risk_type, image_safe, image_safety_label, category, subcategory, raw_response in zip(
        idx, prompts, types, image_safe_values, image_safety_labels, categories, subcategories, raw_responses
    ):
        refused = utils.count_refused_responses(raw_response)
        row = {
            'id': sample_id,
            'prompt': prompt,
            'type': risk_type,
            'image_safe': image_safe,
            'image_safety_label': image_safety_label,
            'category': category,
            'subcategory': subcategory,
            'raw_response': raw_response,
            'refused': refused,
        }
        report.append(row)
        full_refused += refused

    acc = full_refused / len(idx) * 100 if len(idx) else 0
    return report, round(acc, 2)


def create_report_SIUO(**results_dict):
    idx = results_dict["idx"]
    prompts = results_dict["prompt"]
    images = results_dict["image"]
    categories = results_dict["category"]
    safety_warnings = results_dict["safety_warning"]
    reference_answers = results_dict["reference_answer"]
    raw_responses = results_dict["raw_responses"]

    report = []
    full_refused = 0
    for sample_id, prompt, image, category, safety_warning, reference_answer, raw_response in zip(
        idx, prompts, images, categories, safety_warnings, reference_answers, raw_responses
    ):
        refused = utils.count_refused_responses(raw_response)
        row = {
            'id': sample_id,
            'prompt': prompt,
            'image': image,
            'category': category,
            'safety_warning': safety_warning,
            'reference_answer': reference_answer,
            'raw_response': raw_response,
            'refused': refused,
        }
        report.append(row)
        full_refused += refused

    acc = full_refused / len(idx) * 100 if len(idx) else 0
    return report, round(acc, 2)


def get_multi_choice_info(options):
    """
    Given the list of options for multiple choice question
    Return the index2ans and all_choices
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/data_utils.py#L54
    """

    start_chr = "A"
    all_choices = []
    index2ans = {}
    for i, option in enumerate(options):
        index2ans[chr(ord(start_chr) + i)] = option
        all_choices.append(chr(ord(start_chr) + i))

    return index2ans, all_choices

def parse_multi_choice_response(response, all_choices, index2ans):
    """
    Parse the prediction from the generated response.
    Return the predicted index, e.g., A, B, C, D.
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L10
    """
    last_answer_pos = response.rfind("Answer:")
    if last_answer_pos != -1:
        # Extract the string after "Answer:"
        answer_str = response[last_answer_pos + len("Answer:"):].strip()
        
        # Find a unique match in the options
        matching_options = [option for option in all_choices if option in answer_str]
        
        # If a unique match is found, return that option
        if len(matching_options) == 1:
            return matching_options[0]

        
    if isinstance(response, str):
        for char in [",", ".", "!", "?", ";", ":", "'"]:
            response = response.strip(char)
        response = " " + response + " "  # add space to avoid partial match
    else:
        print (response)
        response = ""
    

    index_ans = True
    ans_with_brack = False
    candidates = []
    for choice in all_choices:  # e.g., (A) (B) (C) (D)
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A B C D
            if f"{choice} " in response:
                candidates.append(choice)

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A. B. C. D.
            if f"{choice}." in response:
                candidates.append(choice)

    # if all above doesn't get candidates, check if the content is larger than 5 tokens and try to parse the example
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False  # it's content ans.

    if len(candidates) == 0:  # still not get answer, randomly choose one.
        pred_index = random.choice(all_choices)
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack:
                for can in candidates:
                    index = response.rfind(f"({can})")
                    start_indexes.append(index)  # -1 will be ignored anyway
                # start_indexes = [generated_response.index(f'({can})') for can in candidates]
            else:
                for can in candidates:
                    index = response.rfind(f" {can} ")
                    start_indexes.append(index)
        else:
            for can in candidates:
                index = response.lower().rfind(index2ans[can].lower())
                start_indexes.append(index)
        # get the last one
        pred_index = candidates[np.argmax(start_indexes)]
    else:  # if only one candidate, use it.
        pred_index = candidates[0]

    return pred_index

# DATA PROCESSING -- MMMU


# Process OPEN for MMMU

def check_is_number(string):
    """
    Check if the given string a number.
    """
    try:
        float(string.replace(',', ''))
        return True
    except ValueError:
        # check if there's comma inside
        return False

def normalize_str(string):
    """
    Normalize the str to lower case and make them float numbers if possible.
    """
    # check if characters in the string

    # if number, numerize it.
    string = string.strip()

    is_number = check_is_number(string)

    if is_number:
        string = string.replace(',', '')
        string = float(string)
        # leave 2 decimal
        string = round(string, 2)
        return [string]
    else: # it's likely to be a string
        # lower it 
        string = string.lower()
        if len(string) == 1:
            return [" " + string, string + " "] # avoid trivial matches
        return [string]

def extract_numbers(string):
    """
    Exact all forms of numbers from a string with regex.
    """
    # Pattern for numbers with commas
    pattern_commas = r'-?\b\d{1,3}(?:,\d{3})+\b'
    # Pattern for scientific notation
    pattern_scientific = r'-?\d+(?:\.\d+)?[eE][+-]?\d+'
    # Pattern for simple numbers without commas
    pattern_simple = r'-?(?:\d+\.\d+|\.\d+|\d+\b)(?![eE][+-]?\d+)(?![,\d])'

    # Extract numbers with commas
    numbers_with_commas = re.findall(pattern_commas, string)
    # Extract numbers in scientific notation
    numbers_scientific = re.findall(pattern_scientific, string)
    # Extract simple numbers without commas
    numbers_simple = re.findall(pattern_simple, string)

    # Combine all extracted numbers
    all_numbers = numbers_with_commas + numbers_scientific + numbers_simple
    return all_numbers

def parse_open_response(response):
    """
    Parse the prediction from the generated response.
    Return a list of predicted strings or numbers.
    """
    # content = content.strip("\n").strip(".").strip(" ")
    def get_key_subresponses(response):
        key_responses = []
        response = response.strip().strip(".").lower()
        sub_responses = re.split(r'\.\s(?=[A-Z])|\n', response)
        indicators_of_keys = ['could be ', 'so ', 'is ',
                            'thus ', 'therefore ', 'final ', 'answer ', 'result ']
        key_responses = []
        for index, resp in enumerate(sub_responses):
            # if last one, accept it's an equation (the entire response can be just one sentence with equation)
            if index == len(sub_responses) - 1:
                indicators_of_keys.extend(['='])
            shortest_key_response = None # the shortest response that may contain the answer (tail part of the response)
            for indicator in indicators_of_keys:
                if indicator in resp:
                    if not shortest_key_response:
                        shortest_key_response = resp.split(indicator)[-1].strip()
                    else:
                        if len(resp.split(indicator)[-1].strip()) < len(shortest_key_response):
                            shortest_key_response = resp.split(indicator)[-1].strip()
                    # key_responses.append(resp.split(indicator)[1].strip())
            
            if shortest_key_response:
                # and it's not trivial
                if shortest_key_response.strip() not in [":", ",", ".", "!", "?", ";", ":", "'"]:
                    key_responses.append(shortest_key_response)
        if len(key_responses) == 0: # did not found any
            return [response]
        return key_responses
    
    # pdb.set_trace()
    key_responses = get_key_subresponses(response)

    pred_list = key_responses.copy() # keep the original string response
    for resp in key_responses:
        pred_list.extend(extract_numbers(resp))

    tmp_pred_list = []
    for i in range(len(pred_list)):
        tmp_pred_list.extend(normalize_str(pred_list[i]))
    pred_list = tmp_pred_list

    # remove duplicates
    pred_list = list(set(pred_list))

    return pred_list

def eval_open(gold_i, pred_i):
    """
    Evaluate an open question instance
    """
    correct = False
    if isinstance(gold_i, list):
        # use float to avoid trivial matches
        norm_answers = []
        for answer in gold_i:
            norm_answers.extend(normalize_str(answer))
    else:
        norm_answers = normalize_str(gold_i)
    for pred in pred_i: # pred is already normalized in parse response phase
        if isinstance(pred, str): # if it's a string, then find if ans in the pred_i
            for norm_ans in norm_answers:
                # only see if the string answer in the string pred
                if isinstance(norm_ans, str) and norm_ans in pred:
                    if not correct:
                        correct = True
                    break
        else: # it's a float number
            if pred in norm_answers:
                if not correct:
                    correct = True
                break
    return correct
