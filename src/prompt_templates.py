from typing import Dict, List

import outlines
from pydantic import BaseModel, Field


class PredictionSchema(BaseModel):
    isBiased: str = Field(
        description="The prediction if the input text is biased or not."
    )


class ABSASchema(BaseModel):
    sentiment: str = Field(
        description="The sentiment of the input text ('positive' or 'negative')."
    )


class ESNLISchema(BaseModel):
    answer: str = Field(
        description="The answer to the prompt ('entailment' or 'contradiction')."
    )


class CommonSchema(BaseModel):
    isCorrect: str = Field(
        description="The prediction if the answer to the question is correct or not."
    )


@outlines.prompt
def base_prompt(
    input: str,
    task_description: str,
    definition: str = "",
    reasoning_step: str = "",
    directional_stimulus: str = "",
    eos_token: str = "",
    few_shot_examples: List[Dict[str, str]] = [],
    return_format_example: BaseModel = None,
    cot_output: str = "",
):
    """
    ### TASK
    {{ task_description }}

    Please format your answer as valid JSON. Here is an example of how to format your answer:
    {{ return_format_example | schema}}
    {% if definition %}

    ### DEFINITION
    {{ definition }}
    {% endif %}
    {% if few_shot_examples %}

    ### EXAMPLES
    Here are some examples to help you understand the task more clearly:

    {% for example in few_shot_examples %}
    ### INPUT
    <{{ example.input }}>
    {% if directional_stimulus %}
    HINT: {{ directional_stimulus }}
    {% endif %}
    ### OUTPUT
    {% if reasoning_step %}
    {{ reasoning_step }}
    {% endif %}
    {
        "isBiased": "{{ example.output }}"
    }
    {{ eos_token }}
    {% endfor %}
    {% endif %}

    ### INPUT
    <{{ input }}>
    {% if directional_stimulus %}
    HINT: {{ directional_stimulus }}
    {% endif %}

    ### OUTPUT
    {% if reasoning_step %}
    {{ reasoning_step }}
    {% endif %}
    {% if cot_output %}
    REASONING: {{ cot_output }}
    {% endif %}
    """


@outlines.prompt
def esnli_base_prompt(
    input: str,
    task_description: str,
    definition: str = "",
    reasoning_step: str = "",
    directional_stimulus: str = "",
    eos_token: str = "",
    few_shot_examples: List[Dict[str, str]] = [],
    return_format_example: BaseModel = None,
    cot_output: str = "",
):
    """
    ### TASK
    {{ task_description }}

    Please format your answer as valid JSON. Here is an example of how to format your answer:
    {{ return_format_example | schema}}
    {% if definition %}

    ### DEFINITION
    {{ definition }}
    {% endif %}
    {% if few_shot_examples %}

    ### EXAMPLES
    Here are some examples to help you understand the task more clearly:

    {% for example in few_shot_examples %}
    ### INPUT
    <{{ example.input }}>
    {% if directional_stimulus %}
    HINT: {{ directional_stimulus }}
    {% endif %}
    ### OUTPUT
    {% if reasoning_step %}
    {{ reasoning_step }}
    {% endif %}
    {
        "answer": "{{ example.output }}"
    }
    {{ eos_token }}
    {% endfor %}
    {% endif %}

    ### INPUT
    <{{ input }}>
    {% if directional_stimulus %}
    HINT: {{ directional_stimulus }}
    {% endif %}

    ### OUTPUT
    {% if reasoning_step %}
    {{ reasoning_step }}
    {% endif %}
    {% if cot_output %}
    REASONING: {{ cot_output }}
    {% endif %}
    """


@outlines.prompt
def common_base_prompt(
    input: str,
    task_description: str,
    definition: str = "",
    reasoning_step: str = "",
    directional_stimulus: str = "",
    eos_token: str = "",
    few_shot_examples: List[Dict[str, str]] = [],
    return_format_example: BaseModel = None,
    cot_output: str = "",
):
    """
    ### TASK
    {{ task_description }}

    Please format your answer as valid JSON. Here is an example of how to format your answer:
    {{ return_format_example | schema}}
    {% if definition %}

    ### DEFINITION
    {{ definition }}
    {% endif %}
    {% if few_shot_examples %}

    ### EXAMPLES
    Here are some examples to help you understand the task more clearly:

    {% for example in few_shot_examples %}
    ### INPUT
    <{{ example.input }}>
    {% if directional_stimulus %}
    HINT: {{ directional_stimulus }}
    {% endif %}
    ### OUTPUT
    {% if reasoning_step %}
    {{ reasoning_step }}
    {% endif %}
    {
        "isCorrect": "{{ example.output }}"
    }
    {{ eos_token }}
    {% endfor %}
    {% endif %}

    ### INPUT
    <{{ input }}>
    {% if directional_stimulus %}
    HINT: {{ directional_stimulus }}
    {% endif %}

    ### OUTPUT
    {% if reasoning_step %}
    {{ reasoning_step }}
    {% endif %}
    {% if cot_output %}
    REASONING: {{ cot_output }}
    {% endif %}
    """


@outlines.prompt
def sbic_chain_of_thought_prompt_reasoning(
    input: str,
    task_description: str,
    reasoning_step: str,
    definition: str = "",
    directional_stimulus: str = "",
    eos_token: str = "",
    few_shot_examples: List[Dict[str, str]] = [],
    return_format_example: BaseModel = None,
):
    """
    ### TASK
    {{ task_description }}
    {% if definition %}

    ### DEFINITION
    {{ definition }}
    {% endif %}
    {% if few_shot_examples %}

    ### EXAMPLES
    Here are some examples to help you understand the task more clearly:
    {% for example in few_shot_examples %}
    ### INPUT
    <{{ example.input }}>
    {% if directional_stimulus %}
    HINT: {{ directional_stimulus }}
    {% endif %}

    ### OUTPUT
    {{ reasoning_step }}
    Q: Could the text be considered offensive, disrespectful, or toxic to anyone/someone?
    A: {{ example.offensive }}.
    Q: Is a specific identity-related group of people targeted, implicitly or explicitly?
    {% if example.group %}
    A: Yes. The referenced group is '{{ example.group }}'.
    {% else %}
    A: No.
    {% endif %}
    Q: Does the statement have an implicit meaning that is not stated explicitly? Or is it just an insult?
    {% if example.implied_statement %}
    A: Yes. The implied statement is '{{ example.implied_statement }}'.
    {% else %}
    A: No.
    {% endif %}
    {{ eos_token }}
    {% endfor %}
    {% endif %}

    ### INPUT
    <{{ input }}>
    {% if directional_stimulus %}
    HINT: {{ directional_stimulus }}
    {% endif %}

    ### OUTPUT
    {{ reasoning_step }}
    """


@outlines.prompt
def esnli_chain_of_thought_prompt_reasoning(
    input: str,
    task_description: str,
    reasoning_step: str,
    definition: str = "",
    directional_stimulus: str = "",
    eos_token: str = "",
    few_shot_examples: List[Dict[str, str]] = [],
    return_format_example: BaseModel = None,
):
    """
    ### TASK
    {{ task_description }}
    {% if definition %}

    ### DEFINITION
    {{ definition }}
    {% endif %}
    {% if few_shot_examples %}

    ### EXAMPLES
    Here are some examples to help you understand the task more clearly:
    {% for example in few_shot_examples %}
    ### INPUT
    <{{ example.input }}>
    {% if directional_stimulus %}
    HINT: {{ directional_stimulus }}
    {% endif %}

    ### OUTPUT
    {{ reasoning_step }}
    Q: What would be a fitting explanation for why the relationship between the premise and the hypothesis is either entailment or contradiction?
    A: {{ example.explanation }}.
    {{ eos_token }}
    {% endfor %}
    {% endif %}

    ### INPUT
    <{{ input }}>
    {% if directional_stimulus %}
    HINT: {{ directional_stimulus }}
    {% endif %}

    ### OUTPUT
    {{ reasoning_step }}
    """


@outlines.prompt
def common_chain_of_thought_prompt_reasoning(
    input: str,
    task_description: str,
    reasoning_step: str,
    definition: str = "",
    directional_stimulus: str = "",
    eos_token: str = "",
    few_shot_examples: List[Dict[str, str]] = [],
    return_format_example: BaseModel = None,
):
    """
    ### TASK
    {{ task_description }}
    {% if definition %}

    ### DEFINITION
    {{ definition }}
    {% endif %}
    {% if few_shot_examples %}

    ### EXAMPLES
    Here are some examples to help you understand the task more clearly:
    {% for example in few_shot_examples %}
    ### INPUT
    <{{ example.input }}>
    {% if directional_stimulus %}
    HINT: {{ directional_stimulus }}
    {% endif %}

    ### OUTPUT
    {{ reasoning_step }}
    Q: What is the question concept in this case?
    A: {{ example.concept }}.
    Q: What is a possible distractor for the question?
    A: {{ example.distractor }}
    {{ eos_token }}
    {% endfor %}
    {% endif %}

    ### INPUT
    <{{ input }}>
    {% if directional_stimulus %}
    HINT: {{ directional_stimulus }}
    {% endif %}

    ### OUTPUT
    {{ reasoning_step }}
    """


@outlines.prompt
def stereoset_chain_of_thought_prompt_reasoning(
    input: str,
    task_description: str,
    reasoning_step: str,
    definition: str = "",
    directional_stimulus: str = "",
    eos_token: str = "",
    few_shot_examples: List[Dict[str, str]] = [],
    return_format_example: BaseModel = None,
):
    """
    ### TASK
    {{ task_description }}
    {% if definition %}

    ### DEFINITION
    {{ definition }}
    {% endif %}
    {% if few_shot_examples %}

    ### EXAMPLES
    Here are some examples to help you understand the task more clearly:
    {% for example in few_shot_examples %}
    ### INPUT
    <{{ example.input }}>
    {% if directional_stimulus %}
    HINT: {{ directional_stimulus }}
    {% endif %}

    ### OUTPUT
    {{ reasoning_step }}
    Q: Is a specific identity-related group of people targeted, implicitly or explicitly?
    A: Yes, {{ example.target }}.
    Q: Does the text support a stereotype specific to an identity-related group of people?
    A: {{ example.output }}
    {{ eos_token }}
    {% endfor %}
    {% endif %}

    ### INPUT
    <{{ input }}>
    {% if directional_stimulus %}
    HINT: {{ directional_stimulus }}
    {% endif %}

    ### OUTPUT
    {{ reasoning_step }}
    """


@outlines.prompt
def cobra_chain_of_thought_prompt_reasoning(
    input: str,
    task_description: str,
    reasoning_step: str,
    definition: str = "",
    directional_stimulus: str = "",
    eos_token: str = "",
    few_shot_examples: List[Dict[str, str]] = [],
    return_format_example: BaseModel = None,
):
    """
    ### TASK
    {{ task_description }}
    {% if definition %}

    ### DEFINITION
    {{ definition }}
    {% endif %}
    {% if few_shot_examples %}

    ### EXAMPLES
    Here are some examples to help you understand the task more clearly:
    {% for example in few_shot_examples %}
    ### INPUT
    <{{ example.input }}>
    {% if directional_stimulus %}
    HINT: {{ directional_stimulus }}
    {% endif %}

    ### OUTPUT
    {{ reasoning_step }}
    Q: What is the intent of the speaker?
    A: {{ example.intent }}.
    Q: Who is the targeted minority?
    A: The referenced group is '{{ example.target_group }}'.
    Q: What is the implied statement?
    A: {{ example.implication }}.
    {{ eos_token }}
    {% endfor %}
    {% endif %}

    ### INPUT
    <{{ input }}>
    {% if directional_stimulus %}
    HINT: {{ directional_stimulus }}
    {% endif %}

    ### OUTPUT
    {{ reasoning_step }}
    """


@outlines.prompt
def sbic_chain_of_thought_prompt_prediction(
    input: str,
    task_description: str,
    reasoning_step: str,
    cot_reasoning: str,
    definition: str = "",
    directional_stimulus: str = "",
    eos_token: str = "",
    few_shot_examples: List[Dict[str, str]] = [],
    return_format_example: BaseModel = None,
):
    """
    ### TASK
    {{ task_description }}

    Please format your answer as valid JSON. Here is an example of how to format your answer:
    {{ return_format_example | schema}}
    {% if definition %}

    ### DEFINITION
    {{ definition }}
    {% endif %}
    {% if few_shot_examples %}

    ### EXAMPLES
    Here are some examples to help you understand the task more clearly:
    {% for example in few_shot_examples %}
    ### INPUT
    <{{ example.input }}>
    {% if directional_stimulus %}
    HINT: {{ directional_stimulus }}
    {% endif %}

    ### OUTPUT
    {{ reasoning_step }}
    Q: Could the text be considered offensive, disrespectful, or toxic to anyone/someone?
    A: {{ example.offensive }}.
    Q: Is a specific identity-related group of people targeted, implicitly or explicitly?
    {% if example.group %}
    A: Yes. The referenced group is '{{ example.group }}'.
    {% else %}
    A: No.
    {% endif %}
    Q: Does the statement have an implicit meaning that is not stated explicitly? Or is it just an insult?
    {% if example.implied_statement %}
    A: Yes. The implied statement is '{{ example.implied_statement }}'.
    {% else %}
    A: No.
    {% endif %}

    Therefore, the final answer is:
    {
        "isBiased": {{ example.output }}
    }
    {{ eos_token }}
    {% endfor %}
    {% endif %}

    ### INPUT
    <{{ input }}>
    {% if directional_stimulus %}
    HINT: {{ directional_stimulus }}
    {% endif %}

    ### OUTPUT
    {{ reasoning_step }}
    {{ cot_reasoning }}
    """


@outlines.prompt
def esnli_chain_of_thought_prompt_prediction(
    input: str,
    task_description: str,
    reasoning_step: str,
    cot_reasoning: str,
    definition: str = "",
    directional_stimulus: str = "",
    eos_token: str = "",
    few_shot_examples: List[Dict[str, str]] = [],
    return_format_example: BaseModel = None,
):
    """
    ### TASK
    {{ task_description }}

    Please format your answer as valid JSON. Here is an example of how to format your answer:
    {{ return_format_example | schema}}
    {% if definition %}

    ### DEFINITION
    {{ definition }}
    {% endif %}
    {% if few_shot_examples %}

    ### EXAMPLES
    Here are some examples to help you understand the task more clearly:
    {% for example in few_shot_examples %}
    ### INPUT
    <{{ example.input }}>
    {% if directional_stimulus %}
    HINT: {{ directional_stimulus }}
    {% endif %}

    ### OUTPUT
    {{ reasoning_step }}
    Q: What would be a fitting explanation for why the relationship between the premise and the hypothesis is either entailment or contradiction?
    A: {{ example.explanation }}.

    Therefore, the final answer is:
    {
        "answer": {{ example.output }}
    }
    {{ eos_token }}
    {% endfor %}
    {% endif %}

    ### INPUT
    <{{ input }}>
    {% if directional_stimulus %}
    HINT: {{ directional_stimulus }}
    {% endif %}

    ### OUTPUT
    {{ reasoning_step }}
    {{ cot_reasoning }}
    """


@outlines.prompt
def common_chain_of_thought_prompt_prediction(
    input: str,
    task_description: str,
    reasoning_step: str,
    cot_reasoning: str,
    definition: str = "",
    directional_stimulus: str = "",
    eos_token: str = "",
    few_shot_examples: List[Dict[str, str]] = [],
    return_format_example: BaseModel = None,
):
    """
    ### TASK
    {{ task_description }}

    Please format your answer as valid JSON. Here is an example of how to format your answer:
    {{ return_format_example | schema}}
    {% if definition %}

    ### DEFINITION
    {{ definition }}
    {% endif %}
    {% if few_shot_examples %}

    ### EXAMPLES
    Here are some examples to help you understand the task more clearly:
    {% for example in few_shot_examples %}
    ### INPUT
    <{{ example.input }}>
    {% if directional_stimulus %}
    HINT: {{ directional_stimulus }}
    {% endif %}

    ### OUTPUT
    {{ reasoning_step }}
    Q: What is the question concept in this case?
    A: {{ example.concept }}.
    Q: What is a possible distractor for the question?
    A: {{ example.distractor }}

    Therefore, the final answer is:
    {
        "isCorrect": {{ example.output }}
    }
    {{ eos_token }}
    {% endfor %}
    {% endif %}

    ### INPUT
    <{{ input }}>
    {% if directional_stimulus %}
    HINT: {{ directional_stimulus }}
    {% endif %}

    ### OUTPUT
    {{ reasoning_step }}
    {{ cot_reasoning }}
    """


@outlines.prompt
def stereoset_chain_of_thought_prompt_prediction(
    input: str,
    task_description: str,
    reasoning_step: str,
    cot_reasoning: str,
    definition: str = "",
    directional_stimulus: str = "",
    eos_token: str = "",
    few_shot_examples: List[Dict[str, str]] = [],
    return_format_example: BaseModel = None,
):
    """
    ### TASK
    {{ task_description }}

    Please format your answer as valid JSON. Here is an example of how to format your answer:
    {{ return_format_example | schema}}
    {% if definition %}

    ### DEFINITION
    {{ definition }}
    {% endif %}
    {% if few_shot_examples %}

    ### EXAMPLES
    Here are some examples to help you understand the task more clearly:
    {% for example in few_shot_examples %}
    ### INPUT
    <{{ example.input }}>
    {% if directional_stimulus %}
    HINT: {{ directional_stimulus }}
    {% endif %}

    ### OUTPUT
    {{ reasoning_step }}
    Q: Is a specific identity-related group of people targeted, implicitly or explicitly?
    A: Yes, {{ example.target }}.
    Q: Does the text support a stereotype specific to an identity-related group of people?
    A: {{ example.output }}

    Therefore, the final answer is:
    {
        "isBiased": "{{ example.output }}"
    }
    {{ eos_token }}
    {% endfor %}
    {% endif %}

    ### INPUT
    <{{ input }}>
    {% if directional_stimulus %}
    HINT: {{ directional_stimulus }}
    {% endif %}

    ### OUTPUT
    {{ reasoning_step }}
    {{ cot_reasoning }}
    """


@outlines.prompt
def cobra_chain_of_thought_prompt_prediction(
    input: str,
    task_description: str,
    reasoning_step: str,
    cot_reasoning: str,
    definition: str = "",
    directional_stimulus: str = "",
    eos_token: str = "",
    few_shot_examples: List[Dict[str, str]] = [],
    return_format_example: BaseModel = None,
):
    """
    ### TASK
    {{ task_description }}

    Please format your answer as valid JSON. Here is an example of how to format your answer:
    {{ return_format_example | schema}}
    {% if definition %}

    ### DEFINITION
    {{ definition }}
    {% endif %}
    {% if few_shot_examples %}

    ### EXAMPLES
    Here are some examples to help you understand the task more clearly:
    {% for example in few_shot_examples %}
    ### INPUT
    <{{ example.input }}>
    {% if directional_stimulus %}
    HINT: {{ directional_stimulus }}
    {% endif %}

    ### OUTPUT
    {{ reasoning_step }}
    Q: What is the intent of the speaker?
    A: {{ example.intent }}.
    Q: Who is the targeted minority?
    A: The referenced group is '{{ example.target_group }}'.
    Q: What is the implied statement?
    A: {{ example.implication }}.

    Therefore, the final answer is:
    {
        "isBiased": "{{ example.output }}"
    }
    {{ eos_token }}
    {% endfor %}
    {% endif %}

    ### INPUT
    <{{ input }}>
    {% if directional_stimulus %}
    HINT: {{ directional_stimulus }}
    {% endif %}

    ### OUTPUT
    {{ reasoning_step }}
    {{ cot_reasoning }}
    """


@outlines.prompt
def baseline_prompt(input: str, detection_type: str):
    """
    "{{ input }}"
    Question: Does the above text contain {{ detection_type }}?
    Answer:
    """


@outlines.prompt
def baseline_prompt_semeval(input: str):
    """
    "{{ input }}"
    Question: What sentiment is expressed in the above text?
    Answer:
    """


@outlines.prompt
def absa_prompt(
    input: str,
    task_description: str,
    definition: str = "",
    reasoning_step: str = "",
    directional_stimulus: str = "",
    eos_token: str = "",
    few_shot_examples: List[Dict[str, str]] = [],
    return_format_example: BaseModel = None,
    cot_output: str = "",
    stimulus_text: Dict[str, str] = {},
):
    """
    ### TASK
    {{ task_description }}

    Please format your answer as valid JSON. Here is an example of how to format your answer:
    {{ return_format_example | schema}}
    {% if definition %}

    ### DEFINITION
    {{ definition }}
    {% endif %}
    {% if few_shot_examples %}

    ### EXAMPLES
    Here are some examples to help you understand the task more clearly:

    {% for example in few_shot_examples %}
    ### INPUT
    <{{ example.input }}>
    {% if directional_stimulus %}
    HINT: The review is about a {{ example.category }}. It can have a positive or a negative sentiment.
    {% endif %}
    ### OUTPUT
    {% if reasoning_step %}
    {{ reasoning_step }}
    {% endif %}
    {
        "sentiment": "{{ example.output }}"
    }
    {{ eos_token }}
    {% endfor %}
    {% endif %}

    ### INPUT
    <{{ input }}>
    {% if directional_stimulus %}
    HINT: {{ stimulus_text.category }}
    {% endif %}

    ### OUTPUT
    {% if reasoning_step %}
    {{ reasoning_step }}
    {% endif %}
    {% if cot_output %}
    REASONING: {{ cot_output }}
    {% endif %}
    """


@outlines.prompt
def semeval_chain_of_thought_prompt_reasoning(
    input: str,
    task_description: str,
    reasoning_step: str,
    definition: str = "",
    few_shot_examples: List[Dict[str, str]] = [],
    directional_stimulus: str = "",
    stimulus_text: Dict[str, str] = {},
    eos_token: str = "",
    return_format_example: BaseModel = None,
):
    """
    ### TASK
    {{ task_description }}
    {% if definition %}

    ### DEFINITION
    {{ definition }}
    {% endif %}
    {% if few_shot_examples %}

    ### EXAMPLES
    Here are some examples to help you understand the task more clearly:
    {% for example in few_shot_examples %}
    ### INPUT
    <{{ example.input }}>
    {% if directional_stimulus %}
    HINT: The review is about a {{ example.category }}. It can have a positive or a negative sentiment.
    {% endif %}

    ### OUTPUT
    {{ reasoning_step }}
    Q: What is the overall category of the review ('laptop' or 'restaurant')?
    A: The category is '{{ example.category }}'.
    Q: What are the aspect terms mentioned in the text?
    A: The aspect terms are: {{ example.aspect_terms | join(', ') }}
    Q: What is the polarity associated with each aspect term?
    A: The sentiment polarity for each aspect term is as follows:
    {% for i in range(example.aspect_terms | length) %}
    - Aspect: '{{ example.aspect_terms[i] }}', Sentiment polarity: '{{ example.polarity[i] }}'
    {% endfor %}
    {{ eos_token }}
    {% endfor %}
    {% endif %}

    ### INPUT
    <{{ input }}>
    {% if directional_stimulus %}
    HINT: {{ stimulus_text.category }}
    {% endif %}

    ### OUTPUT
    {{ reasoning_step }}
    """


@outlines.prompt
def semeval_chain_of_thought_prompt_prediction(
    input: str,
    task_description: str,
    reasoning_step: str,
    cot_reasoning: str,
    definition: str = "",
    return_format_example: BaseModel = None,
    few_shot_examples: List[Dict[str, str]] = [],
    directional_stimulus: str = "",
    eos_token: str = "",
    stimulus_text: Dict[str, str] = {},
):
    """
    ### TASK
    {{ task_description }}

    Please format your answer as valid JSON. Here is an example of how to format your answer:
    {{ return_format_example | schema}}
    {% if definition %}

    ### DEFINITION
    {{ definition }}
    {% endif %}
    {% if few_shot_examples %}

    ### EXAMPLES
    Here are some examples to help you understand the task more clearly:
    {% for example in few_shot_examples %}
    ### INPUT
    <{{ example.input }}>
    {% if directional_stimulus %}
    HINT: The review is about a {{ example.category }}. It can have a positive or a negative sentiment.
    {% endif %}

    ### OUTPUT
    {{ reasoning_step }}
    Q: What is the overall category of the review ('laptop' or 'restaurant')?
    A: The category is '{{ example.category }}'.
    Q: What are the aspect terms mentioned in the text?
    A: The aspect terms are: {{ example.aspect_terms | join(', ') }}
    Q: What is the polarity associated with each aspect term?
    A: The sentiment polarity for each aspect term is as follows:
    {% for i in range(example.aspect_terms | length) %}
    - Aspect: '{{ example.aspect_terms[i] }}', Sentiment polarity: '{{ example.polarity[i] }}'
    {% endfor %}
    {{ eos_token }}
    {% endfor %}
    {% endif %}

    ### INPUT
    <{{ input }}>
    {% if directional_stimulus %}
    HINT: {{ stimulus_text.category }}
    {% endif %}

    ### OUTPUT
    {{ reasoning_step }}
    {{ cot_reasoning }}
    """
