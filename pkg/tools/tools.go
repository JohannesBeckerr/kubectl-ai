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
	"maps"
	"slices"
	"time"

	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/journal"
	"github.com/google/uuid"
)

func Lookup(name string) Tool {
	return allTools.Lookup(name)
}

var allTools Tools = Tools{
	tools: make(map[string]Tool),
}

func Default() Tools {
	return allTools
}

// RegisterTool makes a tool available to the LLM.
func RegisterTool(tool Tool) {
	allTools.RegisterTool(tool)
}

type Tools struct {
	tools map[string]Tool
}

func (t *Tools) Lookup(name string) Tool {
	return t.tools[name]
}

func (t *Tools) AllTools() []Tool {
	return slices.Collect(maps.Values(t.tools))
}

func (t *Tools) Names() []string {
	names := make([]string, 0, len(t.tools))
	for name := range t.tools {
		names = append(names, name)
	}
	return names
}

func (t *Tools) RegisterTool(tool Tool) {
	if _, exists := t.tools[tool.Name()]; exists {
		panic("tool already registered: " + tool.Name())
	}
	t.tools[tool.Name()] = tool
}

type InvokeToolOptions struct {
	WorkDir string

	Kubeconfig string
}

type ToolRequestEvent struct {
	CallID    string         `json:"id,omitempty"`
	Name      string         `json:"name,omitempty"`
	Arguments map[string]any `json:"arguments,omitempty"`
}

type ToolResponseEvent struct {
	CallID   string `json:"id,omitempty"`
	Response any    `json:"response,omitempty"`
	Error    string `json:"error,omitempty"`
}

// executeAction handles the execution of a single action
func (t *Tools) InvokeTool(ctx context.Context, name string, arguments map[string]any, opt InvokeToolOptions) (any, error) {
	recorder := journal.RecorderFromContext(ctx)

	tool := t.Lookup(name)
	if tool == nil {
		return "", fmt.Errorf("tool %q not recognized", name)
	}

	callID := uuid.NewString()
	recorder.Write(ctx, &journal.Event{
		Timestamp: time.Now(),
		Action:    "tool-request",
		Payload: ToolRequestEvent{
			CallID:    callID,
			Name:      name,
			Arguments: arguments,
		},
	})

	ctx = context.WithValue(ctx, "kubeconfig", opt.Kubeconfig)
	ctx = context.WithValue(ctx, "work_dir", opt.WorkDir)

	response, err := tool.Run(ctx, arguments)

	{
		ev := ToolResponseEvent{
			CallID:   callID,
			Response: response,
		}
		if err != nil {
			ev.Error = err.Error()
		}
		recorder.Write(ctx, &journal.Event{
			Timestamp: time.Now(),
			Action:    "tool-response",
			Payload:   ev,
		})
	}

	return response, nil
}
