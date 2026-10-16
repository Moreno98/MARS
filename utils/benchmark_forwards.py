# This file provides the forward functions across different benchmarks.

from tqdm import tqdm
import ast
import utils.benchmark_utils as bench_utils

def to_python_list(values):
    if hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, tuple):
        values = list(values)
    if not isinstance(values, list):
        values = [values]
    return values

def forward_MMMUPro(model, loader):
    predictions = []
    questions = []
    full_options = []
    full_gt = []
    raw_responses = []
    idx = []
    for id, question, prompt, images, answer, options in tqdm(loader):
        raw_response = model.forward_multi_image(images, prompt[0], max_new_tokens=50, add_system_prompt=True)
        options = ast.literal_eval(options[0])

        index2ans, all_choices = bench_utils.get_multi_choice_info(options)
        pred_answer = bench_utils.parse_multi_choice_response(raw_response[0].strip(), all_choices, index2ans)

        raw_responses.append(raw_response[0])
        predictions.append(pred_answer)
        full_gt.append(answer[0])
        idx.append(id[0])
        questions.append(question[0])
        full_options.append(options)

    results_dict = {
        'idx': idx,
        'questions': questions,
        'full_options': full_options,
        'raw_responses': raw_responses,
        'predictions': predictions,
        'full_gt': full_gt
    }

    report, acc = bench_utils.create_report_MMMUPro(**results_dict)
    return report, acc 

def forward_MMMU(model, loader):
    predictions = []
    questions = []
    full_options = []
    full_gt = []
    raw_responses = []
    idx = []
    full_question_type = []
    for id, question, prompt, images, answer, options, question_type in tqdm(loader):
        add_system_prompt = False
        max_new_tokens = 256
        if question_type[0] == 'multiple-choice':
            add_system_prompt = True
            max_new_tokens = 128

        raw_response = model.forward_multi_image(images, prompt[0], max_new_tokens=max_new_tokens, add_system_prompt=add_system_prompt)
        options = ast.literal_eval(options[0])
        
        if question_type[0] == 'multiple-choice':
            index2ans, all_choices = bench_utils.get_multi_choice_info(options)
            pred_answer = bench_utils.parse_multi_choice_response(raw_response[0].strip(), all_choices, index2ans)
        elif question_type[0] == 'open':
            pred_answer = bench_utils.parse_open_response(raw_response[0].strip())
        else:
            raise ValueError(f"Unknown question type: {question_type[0]}")

        raw_responses.append(raw_response[0])
        predictions.append(pred_answer)
        full_gt.append(answer[0])
        idx.append(id[0])
        questions.append(question[0])
        full_options.append(options if question_type[0] == 'multiple-choice' else "None")
        full_question_type.append(question_type[0])


    results_dict = {
        'idx': idx,
        'questions': questions,
        'full_options': full_options,
        'raw_responses': raw_responses,
        'predictions': predictions,
        'full_gt': full_gt,
        'question_type': full_question_type
    }

    report, acc = bench_utils.create_report_MMMU(**results_dict)
    return report, acc 

def forward_MMStar(model, loader):
    results_dict = {
        'idx': [],
        'questions': [],
        'image': [],
        'answer': [],
        'category': [],
        'l2_category': [],
        'bench': [],
        'raw_responses': [],
    }

    for data in tqdm(loader):
        raw_response = model.forward_generic(
            images=data['image_tensor'],
            prompts=data['question'],
            max_new_tokens=32,
        )

        results_dict['idx'] += to_python_list(data['id'])
        results_dict['questions'] += to_python_list(data['question'])
        results_dict['image'] += to_python_list(data['image'])
        results_dict['answer'] += to_python_list(data['answer'])
        results_dict['category'] += to_python_list(data['category'])
        results_dict['l2_category'] += to_python_list(data['l2_category'])
        results_dict['bench'] += to_python_list(data['bench'])
        results_dict['raw_responses'] += raw_response

    report, acc = bench_utils.create_report_MMStar(**results_dict)
    return report, acc

def forward_MMSafetyBench(model, loader):
    results_dict = {
        'idx': [],
        'prompt': [],
        'raw_responses': [],
    }

    for data in tqdm(loader):
        if 'image' in data.keys():
            ids, questions, images = data['id'], data['question'], data['image']
        else:
            ids, questions, images = data['id'], data['question'], None

        raw_response = model.forward_generic(
            images = images,
            prompts = questions,
            max_new_tokens=256
        )

        results_dict['idx'] += ids
        results_dict['prompt'] += questions
        results_dict['raw_responses'] += raw_response

    report, acc = bench_utils.create_report_MMSafetyBench(**results_dict)
    return report, acc 

def forward_HADES(model, loader):
    results_dict = {
        'idx': [],
        'prompt': [],
        'scenario': [],
        'keywords': [],
        'step': [],
        'category': [],
        'instruction': [],
        'raw_responses': [],
    }

    for data in tqdm(loader):
        ids = data['id']
        questions = data['question']
        images = data['image']
        raw_response = model.forward_generic(
            images=images,
            prompts=questions,
            max_new_tokens=256
        )

        results_dict['idx'] += ids
        results_dict['prompt'] += questions
        results_dict['scenario'] += data['scenario']
        results_dict['keywords'] += data['keywords']
        results_dict['step'] += data['step']
        results_dict['category'] += data['category']
        results_dict['instruction'] += data['instruction']
        results_dict['raw_responses'] += raw_response

    report, acc = bench_utils.create_report_HADES(**results_dict)
    return report, acc 

def forward_VLGuard(model, loader):
    results_dict = {
        'idx': [],
        'prompt': [],
        'image': [],
        'subset': [],
        'safe': [],
        'harmful_category': [],
        'harmful_subcategory': [],
        'raw_responses': [],
    }

    for data in tqdm(loader):
        raw_response = model.forward_generic(
            images=data['image_tensor'],
            prompts=data['question'],
            max_new_tokens=256
        )

        results_dict['idx'] += data['id']
        results_dict['prompt'] += data['question']
        results_dict['image'] += data['image']
        results_dict['subset'] += data['subset']
        results_dict['safe'] += [bool(value) for value in to_python_list(data['safe'])]
        results_dict['harmful_category'] += data['harmful_category']
        results_dict['harmful_subcategory'] += data['harmful_subcategory']
        results_dict['raw_responses'] += raw_response

    report, acc = bench_utils.create_report_VLGuard(**results_dict)
    return report, acc 

def forward_HoliSafe(model, loader):
    results_dict = {
        'idx': [],
        'prompt': [],
        'type': [],
        'image_safe': [],
        'image_safety_label': [],
        'category': [],
        'subcategory': [],
        'raw_responses': [],
    }

    for data in tqdm(loader):
        raw_response = model.forward_generic(
            images=data['image_tensor'],
            prompts=data['question'],
            max_new_tokens=256
        )

        results_dict['idx'] += [int(value) for value in to_python_list(data['id'])]
        results_dict['prompt'] += data['question']
        results_dict['type'] += data['type']
        results_dict['image_safe'] += [bool(value) for value in to_python_list(data['image_safe'])]
        results_dict['image_safety_label'] += [int(value) for value in to_python_list(data['image_safety_label'])]
        results_dict['category'] += data['category']
        results_dict['subcategory'] += data['subcategory']
        results_dict['raw_responses'] += raw_response

    report, acc = bench_utils.create_report_HoliSafe(**results_dict)
    return report, acc 

def forward_SIUO(model, loader):
    results_dict = {
        'idx': [],
        'prompt': [],
        'image': [],
        'category': [],
        'safety_warning': [],
        'reference_answer': [],
        'raw_responses': [],
    }

    for data in tqdm(loader):
        raw_response = model.forward_generic(
            images=data['image_tensor'],
            prompts=data['question'],
            max_new_tokens=256
        )

        results_dict['idx'] += [int(value) for value in to_python_list(data['id'])]
        results_dict['prompt'] += data['question']
        results_dict['image'] += data['image']
        results_dict['category'] += data['category']
        results_dict['safety_warning'] += data['safety_warning']
        results_dict['reference_answer'] += data['reference_answer']
        results_dict['raw_responses'] += raw_response

    report, acc = bench_utils.create_report_SIUO(**results_dict)
    return report, acc 


