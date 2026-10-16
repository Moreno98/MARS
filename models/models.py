from transformers import AutoModelForImageTextToText, Gemma3ForConditionalGeneration, Qwen3VLForConditionalGeneration, AutoProcessor
import torch
from PIL.Image import Image
import utils.utils as utils

class Molmo2:
    def __init__(
        self,
        model_name, 
        cache_activations=False,
        tokens_type=None,
        do_sample=None,
        perc_token_id=None,
        modality=None,
        device='cuda',
    ):
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            dtype=torch.bfloat16,
            device_map="auto"
        ).eval()
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            device_map="auto",
            padding_side='left'
        )
        self.do_sample = do_sample
        self.tokens_type = tokens_type
        self.perc_token_id = perc_token_id
        self.cache_activations = cache_activations
        self.modality = modality
        self.module_to_probe = self.model.model.transformer

    def __call__(self, *args, **kwargs):
        if self.modality == "text":
            return self.forward_text(*args, **kwargs)
        elif self.modality == "image":
            return self.forward_image(*args, **kwargs)
        elif self.modality == "video":
            return self.forward_video(*args, **kwargs)
        else:
            raise ValueError("Unknown modality")
    
    def forward_multi_image(self, images, prompt, max_new_tokens, **kwargs):
        utterance = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }

        for image in images:
            utterance["content"].append({"type": "image", "image": image.squeeze()})

        messages = [utterance]

        inputs = self.processor.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            tokenize=True, 
            return_dict=True, 
            return_tensors="pt",
            padding=True
        ).to(self.model.device, dtype=torch.bfloat16)

        output = self.generate(
            inputs, 
            max_new_tokens=max_new_tokens
        )

        if isinstance(prompt, str):
            responses = [self._parse_answer(text=prompt, response=output[0])]
        elif isinstance(prompt, list):
            responses = []
            for idx, p in enumerate(prompt):
                responses.append(self._parse_answer(text=p, response=output[idx]))
        return responses

    @torch.no_grad() 
    def forward_video(self, videos, text=None, max_new_tokens=256, prompt=None):
        messages = []
        if isinstance(videos, list) or isinstance(videos, tuple):
            idx = 0
            for video in videos:
                messages.append(
                    [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "video",
                                    "video": video,
                                    "fps": 2.0,
                                },
                            ],
                        }
                    ]
                )
                if text is not None:
                    messages[-1][0]['content'].append(
                        {
                            "type": "text",
                            "text": text[idx],
                        }
                    )
                idx += 1

        inputs = self.processor.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            tokenize=True, 
            return_dict=True, 
            return_tensors="pt",
            padding=True
        ).to(self.model.device, dtype=torch.bfloat16)

        output = self.generate(
            inputs, 
            max_new_tokens=max_new_tokens
        )

        responses = []
        for idx, p in enumerate(videos):
            responses.append(self._parse_answer(text=p, response=output[idx]))
        return responses
    
    @torch.no_grad()
    def generate(self, inputs, max_new_tokens):
        with torch.inference_mode():
            if self.do_sample:
                output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=self.processor.tokenizer.eos_token_id, cache_implementation="static", do_sample=True)
            else:
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    cache_implementation="static",
                    do_sample=False,   
                    temperature=None,  # must be None, not 0.0
                    top_p=None,        # must be None
                    top_k=None,        # must be None
                )
        return output
    
    def _parse_answer(self, response, text=None):
        return self.processor.decode(response, skip_special_tokens=True).split(f'assistant\n')[-1]

    def forward_text(self, text=None, max_new_tokens=256, return_full_output=False):
        if isinstance(text, str):
            messages = [
                {
                    "role": "user", "content": [
                        {"type": "text", "text": text},
                    ]
                },
            ]
        elif isinstance(text, list):
            messages = []
            for p in text:
                messages.append(
                    [
                        {
                            "role": "user", "content": [
                                {"type": "text", "text": p},
                            ]
                        },
                    ]
                )
        
        inputs = self.processor.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            tokenize=True, 
            return_dict=True, 
            return_tensors="pt",
            padding=True
        ).to(self.model.device, dtype=torch.bfloat16)

        if self.cache_activations:
            assert isinstance(text, str), 'Batch size must be one when collecting activations.'
            self.add_target_token(inputs)

        output = self.generate(
            inputs, 
            max_new_tokens=max_new_tokens
        )

        if isinstance(text, str):
            responses = [self._parse_answer(text=text, response=output[0])]
        elif isinstance(text, list):
            responses = []
            for idx, p in enumerate(text):
                responses.append(self._parse_answer(text=p, response=output[idx]))
        return responses

    def add_target_token(self, inputs):
        if self.tokens_type == 'post_instruction':
            first_post_token_idx = len(inputs['input_ids'][0]) - 2
            last_post_token_idx = len(inputs['input_ids'][0])
            len_post_sentence = last_post_token_idx - first_post_token_idx
            n_tokens = len_post_sentence*self.perc_token_id 
            self.module_to_probe.refusal_at_token_idx = (first_post_token_idx-1) + int(n_tokens)
        else:
            raise NotImplementedError('Not yet implemented')

    def forward_generic(self, images=None, prompts=None, max_new_tokens=256):
        assert images is not None or prompts is not None, 'One between images or prompt should be defined.'
        assert prompts is None or prompts is not None and isinstance(prompts, list), 'Expected a batch of prompts.'
        assert images is None or images is not None and (isinstance(images, list) or isinstance(images, torch.Tensor)), 'Expected a batch of images.'

        if prompts is not None:
            batch_size = len(prompts)
        elif isinstance(images, list):
            batch_size = len(images)
        else:
            batch_size = images.shape[0]

        messages = [
            [
                {
                    "role": "user",
                    "content": [],
                } 
            ] for i in range(batch_size)
        ]
        if prompts is not None:
            if isinstance(prompts, list):
                for i in range(batch_size):
                    messages[i][0]['content'].append(
                        {"type": "text", "text": prompts[i]}
                    )

        if images is not None:
            if isinstance(images, list):
                for i in range(batch_size):
                    messages[i][0]['content'].append(
                        {"type": "image", "image": images[i]}
                    )
            elif isinstance(images, torch.Tensor):
                if images.ndim == 4:
                    for i in range(batch_size):
                        messages[i][0]['content'].append(
                            {"type": "image", "image": utils.ensure_bhwc(images[i].unsqueeze(0))}
                        )
                else:
                    raise ValueError(f'Error the tensor dim should be 4. Found: {images.ndim}')

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            padding=True,
        ).to(self.model.device, dtype=torch.bfloat16)

        output = self.generate(
            inputs, 
            max_new_tokens=max_new_tokens
        )

        responses = []
        for idx in range(len(prompts)):
            responses.append(self._parse_answer(response=output[idx]))

        return responses

    def forward_image(self, images, text=None, max_new_tokens=256, return_full_output=False, prompt=None):
        if isinstance(images, torch.Tensor):
            # raise NotImplementedError('Not Implemented Yet.')
            messages = []
            for idx in range(images.shape[0]):
                if prompt is not None:
                    messages.append([
                        {
                            "role": "user", "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image", "image": utils.ensure_bhwc(images[idx])},
                            ]
                        },
                    ])
                else:
                    messages.append([
                        {
                            "role": "user", "content": [
                                {"type": "image", "image": utils.ensure_bhwc(images[idx])},
                            ]
                        },
                    ])
        elif isinstance(images, list):
            raise NotImplementedError('Not Implemented Yet.')
        elif isinstance(images, Image):
            if prompt is not None:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text", "text": prompt
                            },
                            {
                                "type": "image",
                                "image": images,
                            },
                        ],
                    }
                ]
            else:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": images,
                            },
                        ],
                    }
                ]
        else:
            raise NotImplementedError('Not implemented yet.')

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(self.model.device, dtype=torch.bfloat16)

        if self.cache_activations:
            assert isinstance(images, Image) or (isinstance(images, torch.Tensor) and images.shape[0] == 1), "Batch size must be one to collect activations."
            self.add_target_token(inputs)

        output = self.generate(
            inputs, 
            max_new_tokens=max_new_tokens
        )

        if isinstance(images, Image):
            responses = [self._parse_answer(response=output[0])]
        elif isinstance(images, torch.Tensor):
            responses = []
            for idx in range(images.shape[0]):
                responses.append(self._parse_answer(response=output[idx]))
        return responses


class InternVL3_5:
    def __init__(
        self,
        model_name, 
        cache_activations=False,
        tokens_type=None,
        do_sample=None,
        perc_token_id=None,
        modality=None,
        device='cuda',
    ):
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).eval().cuda()
        self.processor = AutoProcessor.from_pretrained(model_name, padding_side='left')
        self.do_sample = do_sample
        self.tokens_type = tokens_type
        self.perc_token_id = perc_token_id
        self.cache_activations = cache_activations
        self.modality = modality
        self.module_to_probe = self.model.model.language_model

    def __call__(self, *args, **kwargs):
        if self.modality == "text":
            return self.forward_text(*args, **kwargs)
        elif self.modality == "image":
            return self.forward_image(*args, **kwargs)
        elif self.modality == "video":
            return self.forward_video(*args, **kwargs)
        else:
            raise ValueError("Unknown modality")
    
    @torch.no_grad()
    def generate(self, inputs, max_new_tokens):
        with torch.inference_mode():
            if self.do_sample:
                output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=self.processor.tokenizer.eos_token_id, cache_implementation="static", do_sample=True)
            else:
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    cache_implementation="static",
                    do_sample=False,   
                    temperature=None,  # must be None, not 0.0
                    top_p=None,        # must be None
                    top_k=None,        # must be None
                )
        return output
    
    def _parse_answer(self, response, text=None):
        return self.processor.decode(response, skip_special_tokens=True).split(f'assistant\n')[-1]

    @torch.no_grad() 
    def forward_video(self, videos, text=None, max_new_tokens=256, prompt=None):
        messages = []
        if isinstance(videos, list) or isinstance(videos, tuple):
            idx = 0
            for video in videos:
                messages.append(
                    [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "video",
                                    "url": video,
                                },
                            ],
                        }
                    ]
                )
                if text is not None:
                    messages[-1][0]['content'].append(
                        {
                            "type": "text",
                            "text": text[idx],
                        }
                    )
                idx += 1

        inputs = self.processor.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            processor_kwargs={
                "fps": 2.0,
                "size": {"height": 448, "width": 448}
            }
        ).to(self.model.device)

        output = self.generate(
            inputs, 
            max_new_tokens=max_new_tokens
        )

        responses = []
        for idx in range(len(videos)):
            responses.append(self._parse_answer(response=output[idx]))
        return responses

    def forward_multi_image(self, images, prompt, max_new_tokens, add_system_prompt=False):
        system_prompt = {
            "role": "system",
            "content": [
                {"type": "text", "text": 'Answer directly. DO NOT REASON.'},
            ],
        }

        utterance = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }

        for image in images:
            utterance["content"].append({"type": "image", "image": image.squeeze()})

        if add_system_prompt:
            messages = [system_prompt, utterance]
        else:
            messages = [utterance]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            processor_kwargs={
                "padding": True,
                "enable_thinking": False,
            }
        ).to(self.model.device)

        output = self.generate(
            inputs, 
            max_new_tokens=max_new_tokens
        )

        if isinstance(prompt, str):
            responses = [self._parse_answer(text=prompt, response=output[0])]
        elif isinstance(prompt, list):
            responses = []
            for idx, p in enumerate(prompt):
                responses.append(self._parse_answer(text=p, response=output[idx]))
        return responses
    
    def forward_generic(self, images=None, prompts=None, max_new_tokens=256):
        assert images is not None or prompts is not None, 'One between images or prompt should be defined.'
        assert prompts is None or prompts is not None and isinstance(prompts, list), 'Expected a batch of prompts.'
        assert images is None or images is not None and (isinstance(images, list) or isinstance(images, torch.Tensor)), 'Expected a batch of images.'

        if prompts is not None:
            batch_size = len(prompts)
        elif isinstance(images, list):
            batch_size = len(images)
        else:
            batch_size = images.shape[0]

        messages = [
            [
                {
                    "role": "user",
                    "content": [],
                } 
            ] for i in range(batch_size)
        ]
        if prompts is not None:
            if isinstance(prompts, list):
                for i in range(batch_size):
                    messages[i][0]['content'].append(
                        {"type": "text", "text": prompts[i]}
                    )

        if images is not None:
            if isinstance(images, list):
                for i in range(batch_size):
                    messages[i][0]['content'].append(
                        {"type": "image", "image": images[i]}
                    )
            elif isinstance(images, torch.Tensor):
                if images.ndim == 4:
                    for i in range(batch_size):
                        messages[i][0]['content'].append(
                            {"type": "image", "image": utils.ensure_bhwc(images[i].unsqueeze(0))}
                        )
                else:
                    raise ValueError(f'Error the tensor dim should be 4. Found: {images.ndim}')

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            processor_kwargs={
                "padding": True,
                "enable_thinking": False,
            }
        ).to(self.model.device)

        output = self.generate(
            inputs, 
            max_new_tokens=max_new_tokens
        )

        responses = []
        for idx, p in enumerate(prompts):
            responses.append(self._parse_answer(text=p, response=output[idx]))
        return responses
    
    def forward_text(self, text=None, max_new_tokens=256, return_full_output=False):
        if isinstance(text, str):
            messages = [
                {
                    "role": "user", "content": [
                        {"type": "text", "text": text},
                    ]
                },
            ]
        elif isinstance(text, list):
            messages = []
            for p in text:
                messages.append(
                    [
                        {
                            "role": "user", "content": [
                                {"type": "text", "text": p},
                            ]
                        },
                    ]
                )
        
        inputs = self.processor.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            tokenize=True, 
            return_dict=True, 
            return_tensors="pt",
            processor_kwargs={
                "padding": True,
                "enable_thinking": False
            }
        ).to(self.model.device, dtype=torch.bfloat16)

        if self.cache_activations:
            assert isinstance(text, str), 'Batch size must be one when collecting activations.'
            self.add_target_token(inputs)

        output = self.generate(
            inputs, 
            max_new_tokens=max_new_tokens
        )

        if isinstance(text, str):
            responses = [self._parse_answer(text=text, response=output[0])]
        elif isinstance(text, list):
            responses = []
            for idx, p in enumerate(text):
                responses.append(self._parse_answer(text=p, response=output[idx]))
        return responses

    def add_target_token(self, inputs):
        if self.tokens_type == 'post_instruction':
            first_post_token_idx = len(inputs['input_ids'][0]) - 2
            last_post_token_idx = len(inputs['input_ids'][0])
            len_post_sentence = last_post_token_idx - first_post_token_idx
            n_tokens = len_post_sentence*self.perc_token_id 
            self.module_to_probe.refusal_at_token_idx = (first_post_token_idx-1) + int(n_tokens)
        else:
            raise NotImplementedError('Not yet implemented')

    def forward_image(self, images, text=None, max_new_tokens=256, return_full_output=False, prompt=None):
        if isinstance(images, torch.Tensor):
            # raise NotImplementedError('Not Implemented Yet.')
            messages = []
            for idx in range(images.shape[0]):
                if prompt is not None:
                    messages.append([
                        {
                            "role": "user", "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image", "image": utils.ensure_bhwc(images[idx])},
                            ]
                        },
                    ])
                else:
                    messages.append([
                        {
                            "role": "user", "content": [
                                {"type": "image", "image": images[idx]},
                            ]
                        },
                    ])
        elif isinstance(images, list):
            raise NotImplementedError('Not Implemented Yet.')
        elif isinstance(images, Image):
            if prompt is not None:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text", "text": prompt
                            },
                            {
                                "type": "image",
                                "image": images,
                            },
                        ],
                    }
                ]
            else:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": images,
                            },
                        ],
                    }
                ]
        else:
            raise NotImplementedError('Not implemented yet.')
        
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(self.model.device, dtype=torch.bfloat16)

        if self.cache_activations:
            assert isinstance(images, Image) or (isinstance(images, torch.Tensor) and images.shape[0] == 1), "Batch size must be one to collect activations."
            self.add_target_token(inputs)

        output = self.generate(
            inputs, 
            max_new_tokens=max_new_tokens
        )

        if isinstance(images, Image):
            responses = [self._parse_answer(response=output[0])]
        elif isinstance(images, torch.Tensor):
            responses = []
            for idx in range(images.shape[0]):
                responses.append(self._parse_answer(response=output[idx]))
        return responses

class Gemma3:
    def __init__(
        self,
        model_name, 
        cache_activations=False,
        tokens_type=None,
        do_sample=None,
        perc_token_id=None,
        modality=None,
        device='cuda',
    ):
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            padding_side="left"
        )
        self.do_sample = do_sample
        self.tokens_type = tokens_type
        self.perc_token_id = perc_token_id
        self.cache_activations = cache_activations
        self.modality = modality
        self.module_to_probe = self.model.model.language_model

    def __call__(self, *args, **kwargs):
        if self.modality == "text":
            return self.forward_text(*args, **kwargs)
        elif self.modality == "image":
            return self.forward_image(*args, **kwargs)
        else:
            raise ValueError("Unknown modality")
    
    def generate(self, inputs, max_new_tokens):
        with torch.inference_mode():
            if self.do_sample:
                output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, cache_implementation="static", do_sample=True)
            else:
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    cache_implementation="static",
                    do_sample=False,   
                    temperature=None,  # must be None, not 0.0
                    top_p=None,        # must be None
                    top_k=None,        # must be None
                )
        return output
    
    def _parse_answer(self, response, text=None):
        return self.processor.decode(response, skip_special_tokens=True).split(f'model\n')[-1]

    def forward_text(self, text=None, max_new_tokens=256, return_full_output=False):
        if isinstance(text, str):
            messages = [
                {
                    "role": "user", "content": [
                        {"type": "text", "text": text},
                    ]
                },
            ]
        elif isinstance(text, list):
            messages = []
            for p in text:
                messages.append(
                    [
                        {
                            "role": "user", "content": [
                                {"type": "text", "text": p},
                            ]
                        },
                    ]
                )

        inputs = self.get_inputs(messages)

        if self.cache_activations:
            assert isinstance(text, str), 'Batch size must be one when collecting activations.'
            self.add_target_token(inputs)

        output = self.generate(
            inputs, 
            max_new_tokens=max_new_tokens
        )

        if isinstance(text, str):
            responses = [self._parse_answer(text=text, response=output[0])]
        elif isinstance(text, list):
            responses = []
            for idx, p in enumerate(text):
                responses.append(self._parse_answer(text=p, response=output[idx]))
        return responses

    def forward_generic(self, images=None, prompts=None, max_new_tokens=256):
        assert images is not None or prompts is not None, 'One between images or prompt should be defined.'
        assert prompts is None or prompts is not None and isinstance(prompts, list), 'Expected a batch of prompts.'
        assert images is None or images is not None and (isinstance(images, list) or isinstance(images, torch.Tensor)), 'Expected a batch of images.'

        if prompts is not None:
            batch_size = len(prompts)
        elif isinstance(images, list):
            batch_size = len(images)
        else:
            batch_size = images.shape[0]

        messages = [
            [
                {
                    "role": "user",
                    "content": [],
                } 
            ] for i in range(batch_size)
        ]
        if prompts is not None:
            if isinstance(prompts, list):
                for i in range(batch_size):
                    messages[i][0]['content'].append(
                        {"type": "text", "text": prompts[i]}
                    )

        if images is not None:
            if isinstance(images, list):
                for i in range(batch_size):
                    messages[i][0]['content'].append(
                        {"type": "image", "image": images[i]}
                    )
            elif isinstance(images, torch.Tensor):
                if images.ndim == 4:
                    for i in range(batch_size):
                        messages[i][0]['content'].append(
                            {"type": "image", "image": utils.ensure_bhwc(images[i].unsqueeze(0))}
                        )
                else:
                    raise ValueError(f'Error the tensor dim should be 4. Found: {images.ndim}')

        inputs = self.get_inputs(messages)

        output = self.generate(
            inputs, 
            max_new_tokens=max_new_tokens
        )

        responses = []
        for idx, p in enumerate(prompts):
            responses.append(self._parse_answer(text=p, response=output[idx]))
        return responses
    
    def forward_multi_image(self, images, prompt, max_new_tokens, **kwargs):
        utterance = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }
        for image in images:
            utterance["content"].append({"type": "image", "image": image.squeeze()})

        messages = [utterance]

        inputs = self.get_inputs(messages)

        output = self.generate(
            inputs, 
            max_new_tokens=max_new_tokens
        )

        if isinstance(prompt, str):
            responses = [self._parse_answer(text=prompt, response=output[0])]
        elif isinstance(prompt, list):
            responses = []
            for idx, p in enumerate(prompt):
                responses.append(self._parse_answer(text=p, response=output[idx]))
        return responses
    
    def add_target_token(self, inputs):
        if self.tokens_type == 'post_instruction':
            first_post_token_idx = len(inputs['input_ids'][0]) - 3
            last_post_token_idx = len(inputs['input_ids'][0])
            len_post_sentence = last_post_token_idx - first_post_token_idx
            n_tokens = len_post_sentence*self.perc_token_id 
            self.module_to_probe.refusal_at_token_idx = (first_post_token_idx-1) + int(n_tokens)
        else:
            raise NotImplementedError('Not yet implemented')

    def get_inputs(self, messages):
        return self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            processor_kwargs={
                'padding': True
            }
        ).to(self.model.device)

    def forward_image(self, images, text=None, max_new_tokens=256, return_full_output=False, prompt=None):
        if isinstance(images, torch.Tensor):
            # raise NotImplementedError('Not Implemented Yet.')
            messages = []
            for idx in range(images.shape[0]):
                if prompt is not None:
                    messages.append([
                        {
                            "role": "user", "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image", "image": images[idx]},
                            ]
                        },
                    ])
                else:
                    messages.append([
                        {
                            "role": "user", "content": [
                                {"type": "image", "image": images[idx]},
                            ]
                        },
                    ])
        elif isinstance(images, list):
            raise NotImplementedError('Not Implemented Yet.')
        elif isinstance(images, Image):
            if prompt is not None:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text", "text": prompt
                            },
                            {
                                "type": "image",
                                "image": images,
                            },
                        ],
                    }
                ]
            else:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": images,
                            },
                        ],
                    }
                ]
        else:
            raise NotImplementedError('Not implemented yet.')
        
        inputs = self.get_inputs(messages)

        if self.cache_activations:
            assert isinstance(images, Image) or (isinstance(images, torch.Tensor) and images.shape[0] == 1), "Batch size must be one to collect activations."
            self.add_target_token(inputs)

        output = self.generate(
            inputs, 
            max_new_tokens=max_new_tokens
        )

        if isinstance(images, Image):
            responses = [self._parse_answer(response=output[0])]
        elif isinstance(images, torch.Tensor):
            responses = []
            for idx in range(images.shape[0]):
                responses.append(self._parse_answer(response=output[idx]))
        return responses

class Qwen3_VL:
    def __init__(
        self, 
        model_name, 
        cache_activations=False,
        tokens_type=None,
        do_sample=None,
        perc_token_id=None,
        modality=None,
        device='cuda',
    ):
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            padding_side='left'
        )
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map="auto"
        )
        self.do_sample = do_sample
        self.tokens_type = tokens_type
        self.perc_token_id = perc_token_id
        self.cache_activations = cache_activations
        self.modality = modality
        self.module_to_probe = self.model.model.language_model
    
    def __call__(self, *args, **kwargs):
        if self.modality == "text":
            return self.forward_text(*args, **kwargs)
        elif self.modality == "image":
            return self.forward_image(*args, **kwargs)
        elif self.modality == 'video':
            return self.forward_video(*args, **kwargs)
        else:
            raise ValueError("Unknown modality")

    @torch.no_grad() 
    def forward_multi_image(self, images, prompt, max_new_tokens, add_system_prompt=False):
        system_prompt = {
            "role": "system",
            "content": [
                {"type": "text", "text": 'Answer directly. DO NOT REASON.'},
            ],
        }

        utterance = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }

        for image in images:
            utterance["content"].append({"type": "image", "image": image.squeeze()})

        if add_system_prompt:
            messages = [system_prompt, utterance]
        else:
            messages = [utterance]

        inputs = self.get_inputs(messages)

        generated_ids = self.generate(
            inputs, 
            max_new_tokens=max_new_tokens
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output
    
    def forward_generic(self, images=None, prompts=None, max_new_tokens=256):
        assert images is not None or prompts is not None, 'One between images or prompt should be defined.'
        assert prompts is None or prompts is not None and isinstance(prompts, list), 'Expected a batch of prompts.'
        assert images is None or images is not None and (isinstance(images, list) or isinstance(images, torch.Tensor)), 'Expected a batch of images.'

        if prompts is not None:
            batch_size = len(prompts)
        elif isinstance(images, list):
            batch_size = len(images)
        else:
            batch_size = images.shape[0]

        messages = [
            [
                {
                    "role": "user",
                    "content": [],
                } 
            ] for i in range(batch_size)
        ]
        if prompts is not None:
            if isinstance(prompts, list):
                for i in range(batch_size):
                    messages[i][0]['content'].append(
                        {"type": "text", "text": prompts[i]}
                    )

        if images is not None:
            if isinstance(images, list):
                for i in range(batch_size):
                    messages[i][0]['content'].append(
                        {"type": "image", "image": images[i]}
                    )
            elif isinstance(images, torch.Tensor):
                if images.ndim == 4:
                    for i in range(batch_size):
                        messages[i][0]['content'].append(
                            {"type": "image", "image": utils.ensure_bhwc(images[i].unsqueeze(0))}
                        )
                else:
                    raise ValueError(f'Error the tensor dim should be 4. Found: {images.ndim}')

        inputs = self.get_inputs(messages)

        generated_ids = self.generate(
            inputs, 
            max_new_tokens=max_new_tokens
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output
        
    @torch.no_grad() 
    def forward_image(self, images, text=None, max_new_tokens=256, prompt=None):
        if isinstance(images, torch.Tensor):
            # raise NotImplementedError('Not Implemented Yet.')
            messages = []
            for idx in range(images.shape[0]):
                if prompt is not None:
                    messages.append([
                        {
                            "role": "user", "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image", "image": images[idx].unsqueeze(0)},
                            ]
                        }
                    ])
                else:
                    messages.append([
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "image": images[idx].unsqueeze(0),
                                },
                            ],
                        }
                    ])
        elif isinstance(images, list):
            raise NotImplementedError('Not Implemented Yet.')
        elif isinstance(images, Image):
            if prompt is not None:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text", "text": prompt
                            },
                            {
                                "type": "image",
                                "image": images,
                            },
                        ],
                    }
                ]
            else:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": images,
                            },
                        ],
                    }
                ]
        else:
            raise NotImplementedError('Not implemented yet.')

        # Preparation for inference
        inputs = self.get_inputs(messages)

        if self.cache_activations:
            assert len(inputs['input_ids']) == 1, 'Supported only batch size 1'
            if self.tokens_type == 'post_instruction':
                first_post_token_idx = len(inputs['input_ids'][0]) - 2 
                last_post_token_idx = len(inputs['input_ids'][0])
                len_post_sentence = last_post_token_idx - first_post_token_idx
                n_tokens = len_post_sentence*self.perc_token_id 
                self.module_to_probe.refusal_at_token_idx = (first_post_token_idx-1) + int(n_tokens)
            else:
                raise NotImplementedError("Not implemented yet.")

        generated_ids = self.generate(inputs, max_new_tokens)

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output

    @torch.no_grad() 
    def forward_video(self, videos, text=None, max_new_tokens=256, prompt=None):
        messages = []
        if isinstance(videos, list) or isinstance(videos, tuple):
            idx = 0
            for video in videos:
                messages.append(
                    [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "video",
                                    "video": video,
                                    "fps": 2.0,
                                },
                            ],
                        }
                    ]
                )
                if text is not None:
                    messages[-1][0]['content'].append(
                        {
                            "type": "text",
                            "text": text[idx],
                        }
                    )
                idx += 1

        # Preparation for inference
        inputs = self.get_inputs(messages)

        generated_ids = self.generate(inputs, max_new_tokens)

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output
    
    def generate(self, inputs, max_new_tokens):
        # Inference: Generation of the output
        with torch.inference_mode():
            if self.do_sample:
                generated_ids = self.model.generate(
                    **inputs, 
                    max_new_tokens=max_new_tokens,
                )
            else:
                generated_ids = self.model.generate(
                    **inputs, 
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=None,      # must be None, not 0.0 when do_sample=False
                    top_p=None,            # must be None when do_sample=False
                    top_k=None,            # must be None when do_sample=False
                    num_beams=1,
                    repetition_penalty=1.0,
                )
            return generated_ids

    def get_inputs(self, messages):
        return self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device, non_blocking=True)

    @torch.no_grad()
    def forward_text(self, text, max_new_tokens=256):
        if isinstance(text, list):
            messages = []
            for p in text:
                messages.append(
                    [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": p},
                            ],
                        }
                    ]
                )
        elif isinstance(text, str):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                    ],
                }
            ]
        else:
            raise NotImplementedError('Not implemented yet.')

        # Preparation for inference
        inputs = self.get_inputs(messages)

        if self.cache_activations:
            if self.tokens_type == 'post_instruction':
                first_post_token_idx = len(inputs['input_ids'][0]) - 2 
                last_post_token_idx = len(inputs['input_ids'][0])
                len_post_sentence = last_post_token_idx - first_post_token_idx
                n_tokens = len_post_sentence*self.perc_token_id 
                self.module_to_probe.refusal_at_token_idx = (first_post_token_idx-1) + int(n_tokens)
            else:
                raise NotImplementedError("Not implemented yet.")
        
        # Inference: Generation of the output
        generated_ids = self.generate(inputs, max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output


class Qwen3_5(Qwen3_VL):
    def __init__(
        self, 
        model_name, 
        cache_activations=False,
        tokens_type=None,
        do_sample=None,
        perc_token_id=None,
        modality=None,
        device='cuda',
    ):
        from transformers import Qwen3_5ForConditionalGeneration
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            padding_side='left'
        )
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto"
        )
        self.do_sample = do_sample
        self.tokens_type = tokens_type
        self.perc_token_id = perc_token_id
        self.cache_activations = cache_activations
        self.modality = modality
        self.module_to_probe = self.model.model.language_model

    def get_inputs(self, messages):
        return self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
            processor_kwargs={
                'padding': True
            }
        ).to(self.model.device, non_blocking=True)
    
    def generate(self, inputs, max_new_tokens):
        # Inference: Generation of the output
        with torch.inference_mode():
            if self.do_sample:
                generated_ids = self.model.generate(
                    **inputs, 
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.model.generation_config.eos_token_id,
                )
            else:
                generated_ids = self.model.generate(
                    **inputs, 
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.model.generation_config.eos_token_id,
                    do_sample=False,
                    temperature=None,      # must be None, not 0.0 when do_sample=False
                    top_p=None,            # must be None when do_sample=False
                    top_k=None,            # must be None when do_sample=False
                    num_beams=1,
                    repetition_penalty=1.0,
                )
            return generated_ids

model_configs = {
    "Gemma3-4B": {
        "class": Gemma3,
        "model_name": "models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767",
    },
    "Molmo2-8B": {
        "class": Molmo2,
        "model_name": "models--allenai--Molmo2-8B/snapshots/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b",
    },
    "InternVL3_5_8B": {
        "class": InternVL3_5,
        "model_name": "models--OpenGVLab--InternVL3_5-8B-HF/snapshots/741a7d03020411e666c6109218ab71e08151ef86",
    },
    "Qwen3-VL-8B": {
        "class": Qwen3_VL,
        "model_name": "models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
    },
    "Qwen3-5-9B": {
        "class": Qwen3_5,
        "model_name": "models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a",
    },
}
