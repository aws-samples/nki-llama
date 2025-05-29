import subprocess


def parse_prompts(filepath):
    with open(filepath, 'r') as file:
        arr = file.read().split('\n\n')
    arr = [prompt.strip() for prompt in arr if prompt.strip()]
    return arr


def parse_prompt_data(filepath):
    with open(filepath, 'r') as file:
        content = file.read()

    blocks = content.split('\n')
    if blocks[-1] == '':
        blocks = blocks[0:-1]
    return [block.split(',') for block in blocks]


def main():
    prompts = parse_prompts("prompts.txt")
    prompt_data = parse_prompt_data("prompt_data.txt")
    assert len(prompts) == len(prompt_data)

    mode = "evaluate"

    # Iterate through the prompts
    for i, prompt in enumerate(prompts):
        data = prompt_data[i]
        seq_len = data[2]
        latency = data[3]
        throughput = data[4]

        command = f'python main.py --enable-nki --mode {mode} --seq-len {seq_len} --base-latency {latency} --base-throughput {throughput} --prompt "{prompt}"'
        print(command)
        
        with open(f'prompt{i}_out.txt', 'w') as outfile:
            subprocess.run(command, shell=True, stdout=outfile)

        print("")

if __name__ == "__main__":
    main()
