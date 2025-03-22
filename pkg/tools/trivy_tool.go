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
	"encoding/json"
	"fmt"
	"os"
	"os/exec"

	"github.com/GoogleCloudPlatform/kubectl-ai/gollm"
	"google.golang.org/protobuf/encoding/prototext"
	"k8s.io/klog/v2"
	ksandboxclient "k8s.io/test-infra/experiment/ksandbox/pkg/client"
	ksandboxpb "k8s.io/test-infra/experiment/ksandbox/protocol/ksandbox/v1alpha1"
)

func init() {
	RegisterTool(&ScanImageWithTrivy{})
}

type ScanImageWithTrivy struct {
	// Image is the image to scan
	Image string `json:"image,omitempty"`
}

func (t *ScanImageWithTrivy) Name() string {
	return "scan_image_with_trivy"
}

func (t *ScanImageWithTrivy) Description() string {
	return "Scans a container image for vulnerabilities, using the trivy tool."
}

func (t *ScanImageWithTrivy) FunctionDefinition() *gollm.FunctionDefinition {
	return &gollm.FunctionDefinition{
		Name:        t.Name(),
		Description: t.Description(),
		Parameters: &gollm.Schema{
			Type: gollm.TypeObject,
			Properties: map[string]*gollm.Schema{
				"image": {
					Type:        gollm.TypeString,
					Description: `The name of the container image to scan.`,
				},
			},
			Required: []string{"image"},
		},
	}
}

func (t *ScanImageWithTrivy) Run(ctx context.Context, opts *ExecutionOptions) (any, error) {
	log := klog.FromContext(ctx)

	if err := opts.parseFunctionArgsInto(t); err != nil {
		return nil, err
	}

	if t.Image == "" {
		return nil, fmt.Errorf("image is required")
	}

	args := []string{"trivy", "image", t.Image}
	cmd := exec.CommandContext(ctx, args[0], args[1:]...)
	cmd.Dir = opts.WorkDir
	cmd.Env = os.Environ()

	runInDocker := false
	if runInDocker {
		dockerImage := "kubectlai-agent-trivy:latest"
		dockerArgs := []string{"docker", "run", "--rm", "-w", "/work", "-v", opts.WorkDir + ":/work", dockerImage}
		dockerArgs = append(dockerArgs, args[1:]...)

		log.Info("running trivy in docker", "args", dockerArgs)
		cmd := exec.CommandContext(ctx, dockerArgs[0], dockerArgs[1:]...)
		cmd.Dir = opts.WorkDir
		cmd.Env = os.Environ()
		return executeCommand(cmd)
	}

	runInKube := true
	if runInKube {
		return t.runInKube(ctx)
	}

	return executeCommand(cmd)
}

func (t *ScanImageWithTrivy) runInKube(ctx context.Context) (*ExecResult, error) {
	namespace := "default"
	image := "fake.registry/kubectlai-agent-trivy:dev"
	buildAgentImage := "fake.registry/ksandbox-agent:dev"

	// We assume this is being run on a developer machine (it's a test program),
	// rather than in-cluster.
	usePortForward := true
	c, err := ksandboxclient.NewAgentClient(ctx, namespace, buildAgentImage, image, usePortForward)
	if err != nil {
		return nil, fmt.Errorf("error building agent client: %w", err)
	}
	defer c.Close()

	command := []string{"trivy", "image", t.Image}

	request := &ksandboxpb.ExecuteCommandRequest{
		Command: command,
	}
	response, err := c.ExecuteCommand(ctx, request)
	if err != nil {
		return nil, fmt.Errorf("error executing in buildagent: %w", err)
	}

	fmt.Printf("response: %s", prototext.Format(response))

	return &ExecResult{
		Stdout:   string(response.Stdout),
		Stderr:   string(response.Stderr),
		ExitCode: int(response.ExitCode),
	}, nil
}

func (opts *ExecutionOptions) parseFunctionArgsInto(task any) error {
	j, err := json.Marshal(opts.FunctionArguments)
	if err != nil {
		return fmt.Errorf("converting function parameters to json: %w", err)
	}
	if err := json.Unmarshal(j, task); err != nil {
		return fmt.Errorf("parsing function parameters into %T: %w", task, err)
	}
	return nil
}
