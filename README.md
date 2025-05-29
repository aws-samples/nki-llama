# NKI Llama

📢 Congratulations to the winning teams! Details about the contest track at ASPLOS are available on the blog post [here](https://www.sigarch.org/the-industry-contest-track-at-sigarch-conferences-the-asplos-2025-experience/). The code from the first place team is available [here](https://github.com/thustorage/nki-llama-contest). The second place team is [here](https://github.com/PASAUCMerced/nki-llama). The third place team is [here](https://github.com/corelab-src/nki-llama).

## Getting Started

This repository provides a package containing the PyTorch model of Llama 3.2 1B. This model **can be compiled with AWS Neuron SDK and run on** a **Trainium instance.** The main file in this package is `llama.py` which contains the model implementation in PyTorch.

In the `llama.py` file, we provide an example NKI kernel for the [RMSNorm operation](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/tutorials/rmsnorm.html) and a guide on how to replace its invocation in the model. This replacement serves as an example of a valid use of a NKI kernel in the PyTorch model. Your task is to identify other parts of the model (operators, fused operators, layers, or even the whole model\!) that can be implemented as NKI kernels and replace them in the original model to achieve better performance.

To learn NKI, follow [the official NKI guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/index.html) and various example NKI kernels from the [nki-samples repository](https://github.com/aws-neuron/nki-samples). Another tool to help with optimizing NKI kernels is [NKI autotune](https://github.com/awslabs/nki-autotune).

## Setup Steps

1. Create a Trainium instance with AWS Neuron SDK v2.21 using EC2 with the following settings:
    1. **Name:** optnki-[xxx]
    2. **AMI:** Deep Learning AMI Neuron (Ubuntu 22.04)
    3. **Instance type:** trn1.2xlarge
    4. **Key pair (login):** create a new key pair
    5. **Metadata version [under “Advanced details”]:** V2 only (otherwise, you will encounter a not authorized error)
    6. When connecting to these instances via SSH, use the username of *ubuntu*.
2. Activate the Neuron virtual environment to run inference by running `source /opt/aws_neuronx_venv_pytorch_2_5_nxd_inference/bin/activate`.
3. Clone this repository and run `cd [PATH]/nki-llama` where `[PATH]` is the directory where you have performed the clone.
4. Download the [Llama3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B) model to a `~/models` folder in your root directory. We recommend doing so using the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli). You can install this by running `pip3 install huggingface_hub[cli]`. You will also need to create an [access token](https://huggingface.co/docs/hub/en/security-tokens). 
To download the models, run the following:
    ```
    cd ~
    mkdir models
    huggingface-cli download --token YOURTOKEN meta-llama/Llama-3.2-1B --local-dir /home/ubuntu/models/llama-3.2-1b
    ```
5. [Llama3.2-1B Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) may be more fun to chat with. You can download and use this model as well.
6. To run inference, navigate to `[PATH]/nki-llama` and run `python main.py --mode generate`.

## NKI Kernel Example
The following steps provide an example of how to utilize NKI kernels in the Llama3.2-1B model:

1. Identify the kernel of interest, e.g. RMSNorm, in the PyTorch model to be optimized with NKI. In the NxDI repository, it is implemented in [modules/custom_calls.py](https://github.com/aws-neuron/neuronx-distributed-inference/blob/main/src/neuronx_distributed_inference/modules/custom_calls.py).

    ```
    class CustomRMSNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-6):
            """
            Use this RMSNorm to perform customized rmsnorm on Neuron
            Note: CustomRMSNorm forward method calls target="AwsNeuronRmsNorm"
            """
            super().__init__()
            self.weight = nn.Parameter(ones(hidden_size))
            self.variance_epsilon = eps
    
        def forward(self, hidden_states):
            original_dtype = hidden_states.dtype
    
            hidden_states = hidden_states.to(torch.float32)
            result = RmsNorm.apply(
                hidden_states, self.weight, self.variance_epsilon, len(hidden_states.shape) - 1
            )
    
            return result.to(original_dtype)
    ```

2. Modify or create a new class for the NKI kernel. `nki_rmsnorm_kernel` refers to the NKI RMSNorm kernel. 

    a. Modify the existing class:

    ```
    class CustomRMSNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-6, nki_enabled=False):
            """
            Use this RMSNorm to perform customized rmsnorm on Neuron
            Note: CustomRMSNorm forward method calls target="AwsNeuronRmsNorm"
            """
            super().__init__()
            self.weight = nn.Parameter(ones(hidden_size))
            self.variance_epsilon = eps
            self.nki_enabled = nki_enabled
    
        def forward(self, hidden_states):
            if self.nki_enabled:
                out_tensor = nki_rmsnorm_kernel(hidden_states, self.weight, self.variance_epsilon)
                return out_tensor
            
            original_dtype = hidden_states.dtype
    
            hidden_states = hidden_states.to(torch.float32)
            result = RmsNorm.apply(
                hidden_states, self.weight, self.variance_epsilon, len(hidden_states.shape) - 1
            )
    
            return result.to(original_dtype)
    ```

    b. Create a new class (this is not what was done in this tutorial):

    ```
    class CustomRMSNormNKI(nn.Module):
        def __init__(self, hidden_size, eps=1e-6):
            """
            Use this RMSNorm to perform customized rmsnorm on Neuron
            Note: CustomRMSNorm forward method calls target="AwsNeuronRmsNorm"
            """
            super().__init__()
            self.weight = nn.Parameter(ones(hidden_size))
            self.variance_epsilon = eps
    
        def forward(self, hidden_states):
            out_tensor = nki_rmsnorm_kernel(hidden_states, self.weight, self.variance_epsilon)
            return out_tensor
    ```
1. You may need to add a batch dimension to input tensor(s), e.g. `a_tensor`. Also be aware of uninitialized data.

    ```
    # iy = nl.arange(a_tensor.shape[1])[None, :]
    iy = nl.arange(a_tensor.shape[2])[None, :]
    
    # num_rows = a_tensor.shape[0]
    num_rows = a_tensor.shape[1]
    ```
    
1. If you modified the existing class, update how the class is invoked in the PyTorch model file `llama.py`.

    ```
    ...
        self.input_layernorm = get_rmsnorm_cls()(
            config.hidden_size,
            eps=config.rms_norm_eps,
            nki_enabled=config.neuron_config.nki_enabled,
        )
    self.post_attention_layernorm = get_rmsnorm_cls()(
        config.hidden_size,
        eps=config.rms_norm_eps,
        nki_enabled=config.neuron_config.nki_enabled,
    )
    ```

    If you created a new class, modify where the kernel is invoked in the PyTorch model file `llama.py` (not done in this tutorial).

    ```
    def get_rmsnorm_cls():
        # Initialize to the appropriate implementation of RMSNorm
        # If infer on NXD -> CustomRMSNorm
        # If infer on CPU -> HF_RMSNorm (CustomRMSNorm does not work on CPU)
        # return CustomRMSNorm if parallel_state.model_parallel_is_initialized() else LlamaRMSNorm
        return CustomRMSNormNKI if parallel_state.model_parallel_is_initialized() else LlamaRMSNorm
    ```

1. Run inference on a single prompt using the NKI kernel and the single evaluation mode by running `python main.py --enable-nki --mode evaluate_single`. If you would like to run the model with specific prompts, pass in `--prompt [PROMPTS]` where `[PROMPTS]` is a comma-separated list of prompts.

## Additional Tools

1. **Profiling:** If you would like to profile your implementation in order to get a better understanding of performance bottlenecks and opportunities for optimization, you can use the [Neuron Profiler](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/neuron-sys-tools/neuron-profile-user-guide.html).
2. **Benchmarking:** You can also leverage the [NKI benchmarking API](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/api/generated/nki.benchmark.html) to retrieve execution latency statistics. 

## Submission

Your submission should be a single Python file called `llama.py`. This file should contain implementations of NKI kernels and also modifications to the original model to invoke these NKI kernels. This file should work as a plug-in replacement for the original `llama.py` of the reference PyTorch implementation provided in this repository.

Make your submission here: https://forms.gle/zZKKS6RzKcerf4vH8

## Benchmarks

Submissions will be tested using 25 benchmarks (prompts) with varying context lengths (TBD, but likely 1K \-\> 128K) and batch sizes (TBD, but likely 1-\>4). We have provided 5 prompts in `prompts.txt` with their corresponding metadata (prompt ID, prompt length, recommended sequence length, and baseline latency/throughput) in `prompt_data.txt`. There are 2 methods of testing these prompts:

1. To avoid recompilation per prompt, you can use a global sequence length (we suggest 640) for all prompts. Run `python main.py --enable-nki --mode evaluate_all --seq-len 640`.
2. Alternatively, you can use a unique sequence length for each prompt (suggested sequence lengths are the third entry in each row of `prompt_data.txt`) at the cost of recompiling the model for each prompt. Run `python test.py` to evaluate these prompts in this fashion.

The remaining 20 prompts will be withheld for evaluation. All benchmarks will become publicly available after the contest is complete.

## Evaluation and Scoring

The contest organizers will execute each team's submission across the twenty withheld benchmarks on a dedicated Trainium instance. The submissions will be evaluated on:

1) Accuracy of generated output vs. our reference implementation. Accuracy evaluation will be a binary assessor: Any benchmark that fails an accuracy threshold will result in a score of 0\.   
2) Latency (Time to first token (TTFT))  
3) Throughput measured as output tokens / second  
4) Amount of model written in NKI (measured as NKI FLOPS / total model FLOPS) (will be applied as a scaling factor for (b) and (c)). Note: NKI FLOPs measures the number of multiply-accumulate (MAC) operations.

Rankings will be established by calculating the total normalized number of points per team, where points are normalized against the baseline.

We define **points** as **Accuracy** (binary) **\* Reduced Latency \* Increased Throughput \* (1 + Normalized NKI FLOPS)**, where:

* **Accuracy** = 1 if accuracy matches or exceeds a predetermined threshold, 0 otherwise  
* **Reduced Latency** = Reference implementation TTFT divided by submission TTFT  
* **Increased Throughput** = Submission tokens/sec divided by reference implementation tokens/sec  
* **Normalized NKI FLOPS** = Submission NKI FLOPS divided by total model FLOPS

For example, a submission that is sufficiently accurate, with 10x reduced latency, 2x increased throughput, and 0.85 normalized NKI FLOPS would obtain 1 \* 10 \* 2 \* 1.85 \= 37 points. For reference, the baseline submission would receive a score of 1.

## Presentations

Teams who successfully submit an entry will be invited to present an informal overview of their approach (roughly 10 to 15 minutes) at a special session held on March 30th during the [Workshop & Tutorial](https://www.asplos-conference.org/asplos2025/workshops-and-tutorials/) days.  Winners will be announced later in the week, with full results being released soon after the conference.

## Contest Eligibility

All are welcome to participate in the contest (including teams from academia, industry, and elsewhere) with the exception of the Contest Organizers and employees of the Contest Sponsor. Individuals are prohibited from participating in multiple teams. In order to be eligible for prizes, teams must commit to releasing an open-source version of their implementation prior to ASPLOS 2026\.

## Frequently Asked Questions

To raise a question, please create an issue in this repository, or feel free to reach out to the contest organizers directly.

## Related Work

* TBD

## Contest Organizers

* Emery Berger (Amazon Web Services), [emerydb@amazon.com](mailto:emerydb@amazon.com)
* Aninda Manocha (Amazon Web Services)
* Wei Tang (Amazon Web Services)
* Emily Webber (Amazon Web Services)
* Ziyang Xu (Amazon Web Services)