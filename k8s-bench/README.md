## k8s-bench

`k8s-bench` is a benchmark for assessing the performance of LLM models for kubernetes related tasks.


### Usage

```sh

# build the k8s-bench binary
go build

# evaluate model performance for scale related tasks
./k8s-bench --agent-bin <path/to/kubectl-ai/binary> --task-pattern scale --kubeconfig <path/to/kubeconfig>
```

Running the benchmark will produce results as below:

```sh
Evaluation Results:
==================

Task: scale-deployment
  Provider: gemini
    gemini-2.0-flash-thinking-exp-01-21: true

Task: scale-down-deployment
  Provider: gemini
    gemini-2.0-flash-thinking-exp-01-21: true

```