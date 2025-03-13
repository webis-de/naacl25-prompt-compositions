import os
from enum import Enum

# Get the directory of this settings script
_settings_script_directory = os.path.abspath(os.path.dirname(__file__))

# Load random seeds
with open(f"{_settings_script_directory}/seeds.txt", "r") as f:
    # Only store lines that are not empty (i.e., if the last line is blank)
    RANDOM_SEED = [int(s) for s in f.read().split("\n") if s != ""]

##### PATHS #####
PATHS = {
    "results": "./results/",
    "experiments": "./experiments/",
    "src": "./src/",
    "tests": "./tests/",
    "logs": "./logs/",
    "cache": "./cache/",  # NOTE use intermediate instead
    "outputs": "./outputs/",
    "tmp": "./tmp/",
    "prompt-predictions": "./outputs/prompt-predictions/",
    "intermediate": "./intermediate/",
    "regression-predictions": "./intermediate/regression-model-outputs/",
    "composition-predictions": "./outputs/composition-predictions/",
    "prompt_components": "./src/prompt-components/",
}
DATASET_PATHS = {
    "sbic": "datasets/sbic",
    "stereoset": "datasets/stereoset",
    "cobraframes": "datasets/cobraframes",
    "semeval": "datasets/semeval-2014",
    "esnli": "datasets/esnli",
    "common_qa": "datasets/commonsense_qa",
}
FILE_NAME_TEMPLATES = {
    "results": f"{PATHS['results']}{{prefix}}{{model_name}}_{{components}}.json",
    "logs": f"{PATHS['logs']}{{prefix}}{{script_name}}_{{model_name}}.log",
    "outputs": f"{PATHS['outputs']}{{prefix}}{{model_name}}_{{components}}.csv",
    "prompt-predictions": f"{PATHS['prompt-predictions']}{{prefix}}{{model_name}}_{{components}}.parquet",  # noqa E501
    "regression-predictions": f"{PATHS['regression-predictions']}{{prefix}}{{model_name}}_{{components}}.parquet",  # noqa E501
    "prompt_template": f"{PATHS['tmp']}{{prefix}}{{model_name}}_{{components}}.txt",
    "preprocessed_sbic": f"{PATHS['intermediate']}sbic_{{split}}.parquet",
    "preprocessed_stereoset": f"{PATHS['intermediate']}stereoset_{{split}}.parquet",
    "preprocessed_cobra": f"{PATHS['intermediate']}cobra_{{split}}.parquet",
    "preprocessed_semeval": f"{PATHS['intermediate']}semeval_{{split}}.parquet",
    "preprocessed_esnli": f"{PATHS['intermediate']}esnli_{{split}}.parquet",
    "preprocessed_common_qa": f"{PATHS['intermediate']}commonsense_qa_{{split}}.parquet",
}

##### BASE SETTINGS #####
LOGGING = {
    "log_dir": PATHS["logs"],
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "datefmt": "%Y-%m-%d %H:%M:%S",
    "time_format_file": "%Y-%m-%d_%H-%M-%S",
    "write_mode": "a",
}
VALID_GENERATION_MODES = ["self-consistency", "single-generation"]
VALID_DECODING_STRATEGIES = ["greedy", "beam-search", "top-k", "top-p"]
VALID_COT_MODE = ["few-shot", "zero-shot"]
VALID_ENGINES = ["vllm", "vLLM", "transformers", "transformer"]

##### MODEL SETTINGS #####
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
SENTENCE_TRANSFORMER_MODELS = {
    "all-MiniLM-L12-v2": "models/all-MiniLM-L12-v2",
    "all-mpnet-base-v2": "models/all-mpnet-base-v2",
}
MAX_LOG_PROBS = 32
MAX_SEQUENCE_LENGTH = 256
BIAS_DETECTION_MAX_TOKEN = 8
BIAS_DETECTION_LOG_PROBS = 16
BEAM_TOP_K = 2  # k <= width
BEAM_WIDTH = 4
NUM_SIMILAR_EXAMPLES = 32

##### PROMPT COMPONENTS SETTINGS #####
PROMPT_COMPONENTS_PATHS = {
    "system-prompts": f"{PATHS['prompt_components']}system_prompts.json",
    "definitions": f"{PATHS['prompt_components']}definitions.json",
    "task-descriptions": f"{PATHS['prompt_components']}task_description.json",
    "reasoning-steps": f"{PATHS['prompt_components']}reasoning_steps.json",
    "directional-stimulus": f"{PATHS['prompt_components']}directional_stimulus.json",
}
VALID_PROMPT_COMPONENTS = [
    "task-description-only",
    "definitions",
    "system-prompts",
    "directional-stimulus",
]
FEW_SHOT_COMPONENTS = ["similar-few-shot", "random-few-shot", "category-few-shot"]
NUM_OF_FEW_SHOT_EXAMPLES = {
    "similar_few_shot": 4,  # 4 examples for each label -> 8 examples
    "random_few_shot": 4,  # 4 random examples for each label -> 8 examples
    "category_few_shot": 1,  # 1 example for each category -> 8 examples
}
SBIC_PROMPT_SETTINGS = {
    "description_type": "sbic",
    "definition_type": "sbic_bias",
    "reasoning_step_type": "zero_shot_sbic",
    "stimulus_type": "sbic",
    "system_prompt_type": "personas",
    "system_prompt_name": "sbic",
}
STEREOSET_PROMPT_SETTINGS = {
    "description_type": "stereoset",
    "definition_type": "stereoset_stereotype",
    "reasoning_step_type": "zero_shot_stereoset",
    "stimulus_type": "stereoset",
    "system_prompt_type": "personas",
    "system_prompt_name": "stereoset",
}
COBRA_PROMPT_SETTINGS = {
    "description_type": "cobra_frames",
    "definition_type": "cobra_frames",
    "reasoning_step_type": "zero_shot_cobra_frames",
    "stimulus_type": "cobra_frames",
    "system_prompt_type": "personas",
    "system_prompt_name": "cobra_frames",
}
SEMEVAL_PROMPT_SETTINGS = {
    "description_type": "absa",
    "definition_type": "absa",
    "reasoning_step_type": "zero_shot_absa",
    "stimulus_type": "absa",
    "system_prompt_type": "personas",
    "system_prompt_name": "absa",
}
ESNLI_PROMPT_SETTINGS = {
    "description_type": "esnli",
    "definition_type": "esnli",
    "reasoning_step_type": "zero_shot_esnli",
    "stimulus_type": "esnli",
    "system_prompt_type": "personas",
    "system_prompt_name": "esnli",
}
COMMON_PROMPT_SETTINGS = {
    "description_type": "common_sense_qa",
    "definition_type": "common_sense_qa",
    "reasoning_step_type": "zero_shot_common_sense_qa",
    "stimulus_type": "common_sense_qa",
    "system_prompt_type": "personas",
    "system_prompt_name": "common_sense_qa",
}
RESPONSE_REGEX = {"isBiased": "(yes|no)"}
ABSA_RESPONSE_REGEX = {"sentiment": "(positive|negative)"}
ESNLI_REGEX = {"answer": "(entailment|contradiction)"}
COMMON_REGEX = {"isCorrect": "(yes|no)"}


##### SBIC SETTINGS #####
SBIC_DATA_SPLITS = [
    "train",
    "test",
    "dev",
    "train_sub_split_random",
    "train_sub_split_balanced",
]
NONE_CATEGORY_NAME = "none"
SBIC_RANDOM_SPLIT_SIZE = 5_000


class Binary_bias_labels(Enum):
    POSITIVE = "yes"
    NEGATIVE = "no"


class SBIC_bias_categories(Enum):
    VICTIM = "victim"
    DISABLES = "disabled"
    NO_BIAS = "none"
    RACE = "race"
    CULTURE = "culture"
    SOCIAL = "social"
    BODY = "body"
    GENDER = "gender"


##### Stereoset SETTINGS #####
class Stereoset_bias_categories(Enum):
    NO_BIAS = "none"
    RACE = "race"
    GENDER = "gender"
    PROFESSION = "profession"
    RELIGION = "religion"


##### COBRA FRAMES SETTINGS #####
COBRA_SPLIT_SIZE = 2_000
COBRA_NON_OFFENSIVE_HINTS = ["not", "none", "neutral", "positively", "no"]
COBRA_OFFENSIVE_HINTS = [
    "microaggression",
    "offensive",
    "condescending",
    "insensitive",
    "hate",
    "sexist",
    "racism",
    "racist",
    "macroaggresion",
    "homophobic",
    "microagg",
    "aggression",
    "aggressive",
    "xenophobic",
    "xenophobia",
]
COBRA_STATEMENT_TEMPLATE = 'This is a conversation between "{speakerIdentity}" and "{listenerIdentity}" in "{speechContext}": "{statement}"'  # noqa


class Cobra_target_groups(Enum):
    NONE = "none"
    ASIAN = "asian"
    MUSLIM = "muslim"
    CHINESE = "chinese"
    LGBTQ = "lgbtq"
    NATIVE_AMERICAN = "native_american"
    MENTAL_DIS = "mental_dis"
    MEXICAN = "mexican"
    WOMEN = "women"
    LATINO = "latino"
    BLACK = "black"
    JEWISH = "jewish"
