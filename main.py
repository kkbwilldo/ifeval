import os
import sys
import nltk
import json
import torch
import argparse
import subprocess
import collections
import dataclasses
import instructions_registry

from tqdm import tqdm
from pathlib import Path
from loguru import logger
from datasets import Dataset, load_dataset
from typing import Dict, Optional, Sequence, Union
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


@dataclasses.dataclass
class InputExample:
    """
    입력 형식을 나타내는 데이터 클래스.

    Attributes:
    - key (int): 각 입력 예제의 고유 키.
    - instruction_id_list (list[str]): 프롬프트 ID의 리스트.
    - prompt (str): 프롬프트 문자열.
    - kwargs (list[Dict[str, Optional[Union[str, int]]]]): 추가 매개변수의 리스트
    """
    key: int 
    instruction_id_list: list[str] 
    prompt: str  
    kwargs: list[Dict[str, Optional[Union[str, int]]]]  


@dataclasses.dataclass
class OutputExample:
    """
    출력 형식을 나타내는 데이터 클래스

    Attributes:
    - instruction_id_list (list[str]): 프롬프트 ID의 리스트.
    - prompt (str): 프롬프트 문자열.
    - response (str): 프롬프트에 대한 response.
    - follow_all_instructions (bool): 모든 지시사항을 따랐는지 여부.
    - follow_instruction_list (list[bool]): 각 지시사항을 따랐는지 여부를 나타내는 리스트.
    """
    instruction_id_list: list[str]
    prompt: str
    response: str
    follow_all_instructions: bool
    follow_instruction_list: list[bool]


def setup_logging(log_file):
    """
    로깅 설정을 초기화하고 설정합니다.

    Parameters:
    - log_file (str): 로그를 기록할 파일의 경로.
    """
    # 기존의 모든 로거를 제거
    logger.remove()
    # 로그를 파일에 기록하는 로거를 추가
    logger.add(log_file, format="{message}", level="INFO")
    # 로그를 표준 출력(콘솔)에 기록하는 로거를 추가
    logger.add(sys.stdout, format="{time} {level} {message}", level="INFO")


def log_print(*args, **kwargs):
    """
    로그에 메시지를 출력합니다.

    Parameters:
    - args: 출력할 임의의 인수들.
    - kwargs: 추가적인 키워드 인수들.

    Returns:
    - None
    """
    # log_print 함수는 전달된 모든 인수를 문자열로 변환하여 공백으로 구분한 후, 이를 하나의 메시지로 결합합니다.
    # 결합된 메시지를 logger의 info 레벨로 출력합니다.
    message = " ".join(map(str, args))
    logger.info(message)


def print_report(outputs):
  """
  정확도 점수에 대한 리포트를 출력합니다.

  Parameters:
  - outputs (list): 테스트 결과를 포함하는 객체들의 리스트.
  """

  prompt_total = 0
  prompt_correct = 0
  instruction_total = 0
  instruction_correct = 0

  tier0_total = collections.defaultdict(int)
  tier0_correct = collections.defaultdict(int)

  tier1_total = collections.defaultdict(int)
  tier1_correct = collections.defaultdict(int)

  for example in outputs:
    follow_instruction_list = example.follow_instruction_list
    instruction_id_list = example.instruction_id_list

    # 프롬프트 레벨의 총 개수와 정확한 개수 계산
    prompt_total += 1
    if all(follow_instruction_list):
      prompt_correct += 1

    # 지시사항 레벨의 총 개수와 정확한 개수 계산
    instruction_total += len(instruction_id_list)
    instruction_correct += sum(follow_instruction_list)

    # tier0 지시사항의 총 개수와 정확한 개수 계산
    for instruction_id, followed_or_not in zip(
        instruction_id_list, follow_instruction_list
    ):
      instruction_id = instruction_id.split(":")[0]
      tier0_total[instruction_id] += 1
      if followed_or_not:
        tier0_correct[instruction_id] += 1

    # tier1 지시사항의 총 개수와 정확한 개수 계산
    for instruction_id, followed_or_not in zip(
        instruction_id_list, follow_instruction_list
    ):
      tier1_total[instruction_id] += 1
      if followed_or_not:
        tier1_correct[instruction_id] += 1

  # 프롬프트 레벨의 정확도 출력
  log_print(f"prompt-level: {prompt_correct / prompt_total}")

  # 지시사항 레벨의 정확도 출력
  log_print(f"instruction-level: {instruction_correct / instruction_total}")

  log_print()

  # tier0 지시사항의 정확도 출력
  for instruction_id in sorted(tier0_total.keys()):
    accuracy = tier0_correct[instruction_id] / tier0_total[instruction_id]
    log_print(f"{instruction_id} {accuracy}")

  log_print()

  # tier1 지시사항의 정확도 출력
  for instruction_id in sorted(tier1_total.keys()):
    accuracy = tier1_correct[instruction_id] / tier1_total[instruction_id]
    log_print(f"{instruction_id} {accuracy}")



def convert_kwargs_in_jsonl(input_file_path):
    """
    JSONL 파일에서 'kwargs'를 JSON 형식의 str에서 dict로 변환하여 새로운 파일에 저장합니다.

    이 조정은 Hugging Face datasets 라이브러리가 자동으로 스키마를 확장할 수 있기 때문에 필요합니다. 
    이는 'kwargs'와 같은 가변 스키마 필드를 가진 데이터의 원래 구조를 변경할 수 있습니다.
    'kwargs'를 Hugging Face datasets에 로드하기 전에 dict으로 변환하면 이 문제를 방지할 수 있습니다.

    Parameters:
    - input_file_path (str): 원본 JSONL 파일의 경로.
    """
    modified_data = []

    with open(input_file_path, 'r') as input_file:
        for line in input_file:
            line_data = json.loads(line)
            # 'kwargs'의 value가 문자열 형식의 JSON인 경우
            # (ex. {"kwargs": "{\"param1\": 10, \"param2\": 20}"} -> {"kwargs": {"param1": 10, "param2": 20}})
            if 'kwargs' in line_data and isinstance(line_data['kwargs'], str):
                line_data['kwargs'] = json.loads(line_data['kwargs'])
            modified_data.append(line_data)

    with open(input_file_path, 'w') as output_file:
        for item in modified_data:
            output_file.write(json.dumps(item) + '\n')


def save_as_jsonl(dataset: Dataset, prompt_column: str, response_column: str, output_dir: str) -> None:
    """
    Hugging Face 데이터셋의 특정 열을 JSONL 파일로 저장합니다.

    Parameters:
    - dataset (Dataset): 처리할 Hugging Face 데이터셋.
    - prompt_column (str): 데이터셋에서 프롬프트를 포함하는 열의 이름.
    - response_column (str): 데이터셋에서 response를 포함하는 열의 이름.
    - output_dir (Path): JSONL 파일을 저장할 디렉토리.

    Returns:
    - None
    """
    output_file = os.path.join(output_dir, f'{response_column}.jsonl')
    with open(output_file, 'w') as f:
        for example in dataset:
            # 각 예제에서 프롬프트와 response를 포함하는 JSON 객체 생성
            json_object = {
                "prompt": example[prompt_column],
                "response": example[response_column]
            }
            f.write(json.dumps(json_object) + '\n')  # JSON 객체를 문자열로 변환하여 파일에 기록


def read_prompt_list(input_jsonl_filename):
    """
    jsonl 파일에서 입력을 읽어옵니다.

    Parameters:
    - input_jsonl_filename (str): 입력 jsonl 파일의 이름.

    Returns:
    - list: InputExample 객체들의 리스트.
    """
    inputs = []
    # jsonl 파일을 읽기 모드로 엽니다
    with open(input_jsonl_filename, "r") as f:
        for l in f:
            # 각 줄을 JSON 형식으로 파싱합니다
            example = json.loads(l)
            # 파싱된 데이터를 InputExample 객체로 변환하여 리스트에 추가합니다
            inputs.append(
                InputExample(key=example["key"],
                             instruction_id_list=example["instruction_id_list"],
                             prompt=example["prompt"],
                             kwargs=example["kwargs"]))
    return inputs


def get_prompt_with_template(message: str) -> str:
    """
    주어진 메시지를 템플릿에 맞게 포맷하여 프롬프트를 생성합니다.

    Parameters:
    - message (str): 사용자가 입력한 메시지.

    Returns:
    - str: 포맷된 프롬프트 문자열.
    """
    SYSTEM_PROMPT_TEMPLATE = """
### System:
You are an AI assistant that follows instructions extremely well. Help as much as you can.
### User:
{instruction}
### Assistant:
"""
    # 주어진 메시지를 SYSTEM_PROMPT_TEMPLATE에 삽입하여 포맷된 프롬프트를 생성
    return SYSTEM_PROMPT_TEMPLATE.format(instruction=message)


def read_prompt_to_response_dict(input_jsonl_filename):
    """
    프롬프트와 response를 매칭하는 딕셔너리를 생성합니다.

    Parameters:
    - input_jsonl_filename (str): 입력 jsonl 파일의 이름.

    Returns:
    - dict: 프롬프트를 키로, response를 값으로 갖는 딕셔너리.
    """
    return_dict = {}
    # jsonl 파일을 읽기 모드로 엽니다
    with open(input_jsonl_filename, "r") as f:
        for l in f:
            # 각 줄을 JSON 형식으로 파싱합니다
            example = json.loads(l)
            # 프롬프트를 키로, response를 값으로 딕셔너리에 추가합니다
            return_dict[example["prompt"]] = example["response"]
    return return_dict


def generate_responses(model_name, row, pipeline):
    """
    입력 모델로 response를 생성합니다.

    Parameters:
    - model_name (str): 모델 이름.
    - row (dict): 데이터셋의 행을 나타내는 딕셔너리.
    - pipeline (Pipeline): 모델의 파이프라인 객체.

    Returns:
    - dict: 생성된 텍스트.
    """
    # 모델에 기반한 프롬프트 전처리
    processed_prompt = get_prompt_with_template(row['prompt'])

    try:
        response = pipeline(processed_prompt, use_cache=True)[0]['generated_text']
    except RuntimeError as e:
        response = "Failed Generation"
    return {f'{model_name}_response': response}


def run(model_name, model_pipeline, dataset):
    """
    주어진 모델과 데이터셋을 사용하여 response와 latency를 생성하고 데이터셋에 새로운 열로 추가합니다.

    Parameters:
    - model_name (str): 사용할 모델의 이름.
    - model_pipeline (Pipeline): 모델 파이프라인 객체.
    - dataset (Dataset): 처리할 데이터셋.

    Returns:
    - Dataset: 새로운 response 및 latency 열이 추가된 데이터셋.
    """

    # 새로운 열 이름 준비
    response_column = f'{model_name}_response'
    time_column = f'{model_name}_time'

    # 데이터를 저장할 리스트 초기화
    responses, times = [], []

    # 데이터셋 처리
    for row in tqdm(dataset, desc=f'Processing {model_name}'):
        result = generate_responses(model_name, row, model_pipeline)
        responses.append(result[f'{model_name}_response'])
#         times.append(result[f'{model_name}_time'])
    
    # 새로운 열로 데이터셋 업데이트
    dataset = dataset.add_column(response_column, responses)
#     dataset = dataset.add_column(time_column, times)
    
    return dataset


def write_outputs(output_jsonl_filename, outputs):
  """
  JSONL 파일에 출력 결과를 씁니다.

  Parameters:
  - output_jsonl_filename (str): 출력 결과를 저장할 JSONL 파일의 이름.
  - outputs (list): 출력 결과 객체들의 리스트.
  """
  assert outputs  # 출력 결과가 비어있지 않은지 확인

  with open(output_jsonl_filename, "w") as f:
    for o in outputs:
      # 출력 객체의 속성들을 JSON 형식으로 변환하여 파일에 씀
      f.write(
          json.dumps(
              {
                  attr_name: o.__getattribute__(attr_name)
                  for attr_name in [
                      name for name in dir(o) if not name.startswith("_")
                  ]
              }
          )
      )
      f.write("\n")  # 각 JSON 객체를 개별 줄에 씀



def test_instruction_following_strict(
    inp,
    prompt_to_response,
):
    """
    주어진 response가 지시사항을 따르는지 테스트합니다.

    Parameters:
    - inp (InputExample): 테스트할 입력 예제.
    - prompt_to_response (dict): 프롬프트와 response를 매칭하는 딕셔너리.

    Returns:
    - OutputExample: 지시사항 준수 여부를 포함한 테스트 결과.
    """
    # 프롬프트에 대응하는 response를 가져옴
    response = prompt_to_response[inp.prompt] 
    # 지시사항 ID 리스트를 가져옴 
    instruction_list = inp.instruction_id_list 
    # 각 지시사항의 준수 여부를 저장할 리스트 
    is_following_list = []  

    for index, instruction_id in enumerate(instruction_list):
        # 지시사항 클래스를 레지스트리에서 가져옴
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)
        
        # kwargs가 문자열인 경우 JSON 형식으로 파싱
        if isinstance(inp.kwargs, str):
            lst = json.loads(inp.kwargs)
            instruction.build_description(**lst[index])
        else:
            instruction.build_description(**inp.kwargs[index])
        
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=inp.prompt)
        
        # response가 존재하고 지시사항을 따르는지 확인
        if response.strip() and instruction.check_following(response):
            is_following_list.append(True)
        else:
            is_following_list.append(False)

    # 결과를 OutputExample 객체로 반환
    return OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    )


def test_instruction_following_loose(
    inp,
    prompt_to_response,
):
  """
  지시사항을 따르는지 여부를 느슨하게 테스트합니다.

  Parameters:
  - inp (InputExample): InputExample 객체로, 지시사항 ID 리스트와 프롬프트를 포함합니다.
  - prompt_to_response (dict): 프롬프트와 그에 대한 response를 매핑한 사전.

  Returns:
  - OutputExample: 지시사항을 따르는지 여부에 대한 결과를 포함하는 객체.
  """
  
  response = prompt_to_response[inp.prompt]
  r = response.split("\n")
  
  # response의 첫 번째 줄 제거
  response_remove_first = "\n".join(r[1:]).strip()
  
  # response의 마지막 줄 제거
  response_remove_last = "\n".join(r[:-1]).strip()
  
  # response의 첫 번째 및 마지막 줄 제거
  response_remove_both = "\n".join(r[1:-1]).strip()
  
  # response에서 '*' 제거
  revised_response = response.replace("*", "")
  
  # '*' 제거된 response의 첫 번째 줄 제거
  revised_response_remove_first = response_remove_first.replace("*", "")
  
  # '*' 제거된 response의 마지막 줄 제거
  revised_response_remove_last = response_remove_last.replace("*", "")
  
  # '*' 제거된 response의 첫 번째 및 마지막 줄 제거
  revised_response_remove_both = response_remove_both.replace("*", "")
  
  all_responses = [
      response,
      revised_response,
      response_remove_first,
      response_remove_last,
      response_remove_both,
      revised_response_remove_first,
      revised_response_remove_last,
      revised_response_remove_both,
  ]
  
  instruction_list = inp.instruction_id_list
  is_following_list = []

  for index, instruction_id in enumerate(instruction_list):
    instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
    instruction = instruction_cls(instruction_id)

    # inp.kwargs가 문자열인 경우 JSON으로 파싱
    if isinstance(inp.kwargs, str):
        lst = json.loads(inp.kwargs)
        instruction.build_description(**lst[index])
    else:
        instruction.build_description(**inp.kwargs[index])
    
    args = instruction.get_instruction_args()
    if args and "prompt" in args:
      instruction.build_description(prompt=inp.prompt)

    is_following = False
    for r in all_responses:
      if r.strip() and instruction.check_following(r):
        is_following = True
        break

    is_following_list.append(is_following)

  return OutputExample(
      instruction_id_list=inp.instruction_id_list,
      prompt=inp.prompt,
      response=response,
      follow_all_instructions=all(is_following_list),
      follow_instruction_list=is_following_list,
  )


def parsing():

    parser = argparse.ArgumentParser(description='Evaluate LLMS on Instruction Following task')
    parser.add_argument('--output-dir', type=str, default="/home/kbkim/ifeval/benchmark", help="벤치마크 평가의 결과를 저장할 디렉토리")
    parser.add_argument('--num-data', type=int, default=541, help="벤치마크 데이터셋 중 사용할 데이터 갯수")
    parser.add_argument('--benchmark-dataset', type=str, default='harpreetsahota/Instruction-Following-Evaluation-for-Large-Language-Models', help="사용할 벤치마크 데이터셋")
    parser.add_argument('--max-seq-length', type=int, default=8192, help="모델의 최대 입력 길이")
    parser.add_argument('--input-filename', type=str, default="dataset_for_eval.jsonl", help="입력 파일 이름")
    parser.add_argument('--log-name', type=str, default="evaluate_two_llms.log", help="기록할 로그의 이름")
    parser.add_argument('--model-id', type=str, default="/home/kbkim/checkpoints_llama3_8b_instruct")
    parser.add_argument('--control-model-name', type=str, default="llama3_8b_instruct")
    parser.add_argument('--experimental-model-name', type=str, default="4bit_quantized_llama3_8b_instruct")

    args = parser.parse_args()

    return args


def main(args):

    # nltk의 punkt 다운로드
    nltk.download('punkt')

    # 벤치마크 결과를 저장할 디렉토리 생성
    os.makedirs(args.output_dir,exist_ok=True)

    # 벤치마크 데이터셋 로드
    dataset = load_dataset(args.benchmark_dataset, split='train')
    # 데이터셋의 일부만 사용하여 평가
    dataset = dataset.select(range(args.num_data))
    dataset.to_json(
        os.path.join(args.output_dir,args.input_filename)
    )

    # JSON 형식의 문자열인 kwargs를 dict로 전환
    convert_kwargs_in_jsonl(
        os.path.join(args.output_dir,args.input_filename),
    )

    #####################################################################
    # 원본 모델의 response 수집
    model_id = args.model_id
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=args.max_seq_length,
        return_full_text=False
    )

    # response 생성
    dataset = run(args.control_model_name, model_pipeline, dataset)

    del model
    del model_pipeline
    for _ in range(3):
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    # 양자화된 모델의 response 수집
    model_id = args.model_id
    q_model = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    experimental_model_pipeline = pipeline(
        "text-generation",
        model=q_model,
        tokenizer=tokenizer,
        max_length=args.max_seq_length,
        return_full_text=False
    )

    # response 생성
    dataset = run(args.experimental_model_name, experimental_model_pipeline, dataset)

    # 평가를 위해 데이터셋 저장
    dataset.to_json(os.path.join(args.output_dir,args.input_filename))
    #####################################################################

    #####################################################################
    # responses 평가

    prompt_column = "prompt"
    response_columns = [c+"_response" for c in [args.control_model_name, args.experimental_model_name]]
    
    for response_column in response_columns:
        save_as_jsonl(dataset, prompt_column, response_column, args.output_dir)

    # logger 셋업
    log_file = os.path.join(args.output_dir,args.log_name)
    setup_logging(log_file) 

    # Instruction Following 결과 수집
    for response_column in response_columns:
        response_file = os.path.join(args.output_dir,f"{response_column}.jsonl")

        # JSON 파일에서 입력 읽어오기
        inputs = read_prompt_list(os.path.join(args.output_dir,args.input_filename))
        prompt_to_response = read_prompt_to_response_dict(response_file)

        for func, output_file_name in [
            (test_instruction_following_strict, "eval_results_strict"),
            (test_instruction_following_loose, "eval_results_loose"),
        ]:
            logger.info(f"Generating {output_file_name}...")
            outputs = []
            for inp in inputs:
                outputs.append(func(inp, prompt_to_response))
            follow_all_instructions = [o.follow_all_instructions for o in outputs]
            accuracy = sum(follow_all_instructions) / len(outputs)
            logger.info(f"Accuracy: {accuracy}")

            output_file_name = os.path.join(
                args.output_dir, output_file_name + ".jsonl"
            )
            write_outputs(output_file_name, outputs)
            logger.info(f"Generated: {output_file_name}")

            # Prints instruction following accuracy report.
            log_print("=" * 64)
            log_print(f"{output_file_name} Accuracy Scores:")
            print_report(outputs)


        print(f"Completed evaluation for {response_column}")
    #####################################################################
    
    # 평가 종료 메시지
    print(f"All evaluations complete. Check the log file: {log_file} for details.")

if __name__ == "__main__":
    args = parsing()
    main(args)
