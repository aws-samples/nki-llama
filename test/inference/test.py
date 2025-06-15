import subprocess
import argparse
import json


def parse_prompts(filepath):
    """Parse prompts from JSON file"""
    with open(filepath, 'r') as file:
        data = json.load(file)
    
    # Extract just the prompt text from each prompt object
    prompts = [prompt_obj['prompt'] for prompt_obj in data['prompts']]
    return prompts


def parse_prompt_data(filepath):
    """Parse prompt performance data from JSON file"""
    with open(filepath, 'r') as file:
        data = json.load(file)
    
    # Convert the JSON data to the expected format (list of lists)
    # Each inner list contains: [index, word_count, sequence_length, baseline_latency_ms, baseline_throughput]
    prompt_data = []
    for item in data['prompt_performance_data']:
        prompt_data.append([
            str(item['index']),
            str(item['word_count']),
            str(item['sequence_length']),
            str(item['baseline_latency_ms']),
            str(item['baseline_throughput'])
        ])
    
    return prompt_data

def parse_args():
    parser = argparse.ArgumentParser()

    # repository path
    parser.add_argument("--repository-path", type=str, default="~/nki-llama")
    
    return parser.parse_args()

def main():
    args = parse_args()
    prompts = parse_prompts(f"{args.repository_path}/data/prompts.json")
    prompt_data = parse_prompt_data(f"{args.repository_path}/data/prompt_data.json")
    assert len(prompts) == len(prompt_data)

    mode = "evaluate_single"

    # Iterate through the prompts
    for i, prompt in enumerate(prompts):
        data = prompt_data[i]
        seq_len = data[2]
        latency = data[3]
        throughput = data[4]
 
        command = f'python {args.repository_path}/src/inference/main.py --enable-nki --mode {mode} --seq-len {seq_len} --base-latency {latency} --base-throughput {throughput} --prompt "{prompt}"'
        print(command)
        
        with open(f'prompt{i}_out.txt', 'w') as outfile:
            subprocess.run(command, shell=True, stdout=outfile)

        print("")

if __name__ == "__main__":
    main()
