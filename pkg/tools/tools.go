// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package tools

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"strings"

	"github.com/GoogleCloudPlatform/kubectl-ai/gollm"
)

func All() Tools {
	return allTools
}

func Lookup(name string) Tool {
	return allTools[name]
}

func (t Tools) Names() []string {
	names := make([]string, 0, len(t))
	for name := range t {
		names = append(names, name)
	}
	return names
}

var allTools Tools = Tools{
	"kubectl": &Kubectl{},
	"bash":    &BashTool{},
}

const (
	bashBin = "/bin/bash"
)

type Kubectl struct{}

func (t *Kubectl) Name() string {
	return "kubectl"
}

func (t *Kubectl) Description() string {
	return "Executes a kubectl command against user's Kubernetes cluster. Use this tool only when you need to query or modify the state of user's Kubernetes cluster."
}

func (t *Kubectl) FunctionDefinition() *gollm.FunctionDefinition {
	return &gollm.FunctionDefinition{
		Name:        t.Name(),
		Description: t.Description(),
		Parameters: &gollm.Schema{
			Type: gollm.TypeObject,
			Properties: map[string]*gollm.Schema{
				"command": {
					Type: gollm.TypeString,
					Description: `The complete kubectl command to execute. Please including the kubectl prefix as well.
Example:
user: what pods are running in the cluster?
assistant: kubectl get pods

user: what is the status of the pod my-pod?
assistant: kubectl get pod my-pod -o jsonpath='{.status.phase}'
`,
				},
				"modifies_resource": {
					Type: gollm.TypeString,
					Description: `Whether the command modifies a kubernetes resource.
Possible values:
- "yes" if the command modifies a resource
- "no" if the command does not modify a resource
- "unknown" if the command's effect on the resource is unknown
`,
				},
			},
		},
	}
}

func (t *Kubectl) Run(ctx context.Context, args map[string]any) (any, error) {
	kubectlArgs := KubectlArgs{
		Kubeconfig: ctx.Value("kubeconfig").(string),
		WorkDir:    ctx.Value("work_dir").(string),
		Command:    args["command"].(string),
	}
	return runKubectlCommand(kubectlArgs.Command, kubectlArgs.Kubeconfig, kubectlArgs.WorkDir)
}

type KubectlArgs struct {
	Kubeconfig string `json:"kubeconfig"`
	WorkDir    string `json:"work_dir"`
	Command    string `json:"command"`
}

// runKubectlCommand executes a kubectl command with the specified kubeconfig and returns the output.
func runKubectlCommand(command string, kubeconfig string, workDir string) (string, error) {
	if strings.Contains(command, "kubectl edit") {
		return "interactive mode not supported for kubectl, please use non-interactive commands", nil
	}
	if strings.Contains(command, "kubectl port-forward") {
		return "port-forwarding is not allowed because assistant is running in an unattended mode, please try some other alternative", nil
	}

	cmd := exec.Command(bashBin, "-c", command)
	cmd.Env = os.Environ()
	cmd.Dir = workDir

	if kubeconfig != "" {
		kubeconfig, err := expandShellVar(kubeconfig)
		if err != nil {
			return "", err
		}
		cmd.Env = append(cmd.Env, "KUBECONFIG="+kubeconfig)
	}

	output, err := cmd.CombinedOutput()
	if err != nil {
		if strings.Contains(string(output), "command not found") {
			return "error: command not found. Note that if its a kubectl command, please specify the full command including the kubectl prefix, for example: 'kubectl get pods'", nil
		}
		return string(output), nil
	}
	return string(output), nil
}

// expandShellVar expands shell variables and syntax using bash
func expandShellVar(value string) (string, error) {
	if strings.Contains(value, "~") {
		cmd := exec.Command(bashBin, "-c", fmt.Sprintf("echo %s", value))
		output, err := cmd.Output()
		if err != nil {
			return "", err
		}
		return strings.TrimSpace(string(output)), nil
	}
	return os.ExpandEnv(value), nil
}

type BashTool struct{}

func (t *BashTool) Name() string {
	return "bash"
}

func (t *BashTool) Description() string {
	return "Executes a bash command. Use this tool only when you need to execute a bash command."
}

func (t *BashTool) FunctionDefinition() *gollm.FunctionDefinition {
	return &gollm.FunctionDefinition{
		Name:        t.Name(),
		Description: t.Description(),
		Parameters: &gollm.Schema{
			Type: gollm.TypeObject,
			Properties: map[string]*gollm.Schema{
				"command": {
					Type:        gollm.TypeString,
					Description: `The bash command to execute.`,
				},
				"modifies_resource": {
					Type: gollm.TypeString,
					Description: `Whether the command modifies a kubernetes resource.
Possible values:
- "yes" if the command modifies a resource
- "no" if the command does not modify a resource
- "unknown" if the command's effect on the resource is unknown
`,
				},
			},
		},
	}
}

func (t *BashTool) Run(ctx context.Context, args map[string]any) (any, error) {
	kubeconfig := ctx.Value("kubeconfig").(string)
	workDir := ctx.Value("work_dir").(string)
	command := args["command"].(string)
	return runBashCmd(command, kubeconfig, workDir)
}

// runBashCmd executes a bash command and returns the output
func runBashCmd(command string, kubeconfig string, workDir string) (string, error) {
	if strings.Contains(command, "kubectl edit") {
		return "interactive mode not supported for kubectl, please use non-interactive commands", nil
	}
	if strings.Contains(command, "kubectl port-forward") {
		return "port-forwarding is not allowed because assistant is running in an unattended mode, please try some other alternative", nil
	}

	cmd := exec.Command(bashBin, "-c", command)
	cmd.Dir = workDir
	cmd.Env = os.Environ()
	if kubeconfig != "" {
		kubeconfig, err := expandShellVar(kubeconfig)
		if err != nil {
			return "", err
		}
		cmd.Env = append(cmd.Env, "KUBECONFIG="+kubeconfig)
	}

	output, err := cmd.CombinedOutput()
	if err != nil {
		return string(output), err
	}
	return string(output), nil
}
