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

package main

import (
	"fmt"
	"os"
	"os/exec"
	"strings"
)

var (
	bashBin = "/bin/bash"
)

func buildTools() map[string]func(input string, kubeconfig string, workDir string) (string, error) {
	tools := make(map[string]func(input string, kubeconfig string, workDir string) (string, error))

	tools["kubectl"] = kubectlRunner
	tools["cat"] = bashRunner
	tools["bash"] = bashRunner

	return tools
}

// kubectlRunner executes a kubectl command with the specified kubeconfig and returns the output.
func kubectlRunner(command string, kubeconfig string, workDir string) (string, error) {
	if strings.Contains(command, "kubectl edit") {
		return "interactive mode not supported for kubectl, please use non-interactive commands", nil
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

// bashRunner executes a bash command and returns the output
func bashRunner(command string, kubeconfig string, workDir string) (string, error) {
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
