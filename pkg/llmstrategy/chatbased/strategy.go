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

package chatbased

import (
	"context"
	_ "embed"
	"fmt"
	"html/template"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/GoogleCloudPlatform/kubectl-ai/gollm"
	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/journal"
	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/llmstrategy"
	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/tools"
	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/ui"
	"k8s.io/klog/v2"
)

//go:embed chatbased_systemprompt_template_default.txt
var defaultSystemPromptChatAgent string

type Strategy struct {
	LLM gollm.Client

	// PromptTemplateFile allows specifying a custom template file
	PromptTemplateFile string

	// Recorder captures events for diagnostics
	Recorder journal.Recorder

	RemoveWorkDir bool

	MaxIterations int

	Kubeconfig          string
	AsksForConfirmation bool

	Tools tools.Tools
}

type Conversation struct {
	strategy *Strategy

	// Recorder captures events for diagnostics
	recorder journal.Recorder

	UI      ui.UI
	llmChat gollm.Chat

	workDir string
}

func (s *Strategy) NewConversation(ctx context.Context, u ui.UI) (llmstrategy.Conversation, error) {
	log := klog.FromContext(ctx)

	// Create a temporary working directory
	workDir, err := os.MkdirTemp("", "agent-workdir-*")
	if err != nil {
		log.Error(err, "Failed to create temporary working directory")
		return nil, err
	}

	log.Info("Created temporary working directory", "workDir", workDir)

	systemPrompt, err := s.generatePrompt(ctx, defaultSystemPromptChatAgent)
	if err != nil {
		log.Error(err, "Failed to generate system prompt")
		return nil, err
	}

	// Start a new chat session
	llmChat := s.LLM.StartChat(systemPrompt)

	var functionDefinitions []*gollm.FunctionDefinition
	for _, tool := range s.Tools {
		functionDefinitions = append(functionDefinitions, tool.FunctionDefinition())
	}
	// Sort function definitions to help KV cache reuse
	sort.Slice(functionDefinitions, func(i, j int) bool {
		return functionDefinitions[i].Name < functionDefinitions[j].Name
	})
	if err := llmChat.SetFunctionDefinitions(functionDefinitions); err != nil {
		return nil, fmt.Errorf("setting function definitions: %w", err)
	}

	return &Conversation{
		strategy: s,
		recorder: s.Recorder,
		UI:       u,
		llmChat:  llmChat,
		workDir:  workDir,
	}, nil
}

func (c *Conversation) Close() error {
	if c.workDir != "" {
		if c.strategy.RemoveWorkDir {
			if err := os.RemoveAll(c.workDir); err != nil {
				klog.Warningf("error cleaning up directory %q: %v", c.workDir, err)
			}
		}
	}
	return nil
}

// RunOneRound executes a chat-based agentic loop with the LLM using function calling.
func (a *Conversation) RunOneRound(ctx context.Context, query string) error {
	log := klog.FromContext(ctx)
	log.Info("Starting chat loop for query:", "query", query)

	// currChatContent tracks chat content that needs to be sent
	// to the LLM in each iteration of  the agentic loop below
	var currChatContent []any

	// Set the initial message to start the conversation
	currChatContent = []any{query} //fmt.Sprintf("can you help me with query: %q", query)}

	currentIteration := 0
	maxIterations := a.strategy.MaxIterations

	for currentIteration < maxIterations {
		log.Info("Starting iteration", "iteration", currentIteration)

		a.recorder.Write(ctx, &journal.Event{
			Timestamp: time.Now(),
			Action:    "llm-chat",
			Payload:   []any{currChatContent},
		})

		response, err := a.llmChat.Send(ctx, currChatContent...)
		if err != nil {
			log.Error(err, "Error sending initial message")
			return err
		}

		a.recorder.Write(ctx, &journal.Event{
			Timestamp: time.Now(),
			Action:    "llm-response",
			Payload:   response,
		})

		currChatContent = nil

		if len(response.Candidates()) == 0 {
			log.Error(nil, "No candidates in response")
			return fmt.Errorf("no candidates in LLM response")
		}

		candidate := response.Candidates()[0]

		// Process each part of the response
		var functionCalls []gollm.FunctionCall

		for _, part := range candidate.Parts() {
			// Check if it's a text response
			if text, ok := part.AsText(); ok {
				log.Info("text response", "text", text)
				textResponse := text
				// If we have a text response, render it
				if textResponse != "" {
					a.UI.RenderOutput(ctx, textResponse, ui.RenderMarkdown())
				}
			}

			// Check if it's a function call
			if calls, ok := part.AsFunctionCalls(); ok && len(calls) > 0 {
				log.Info("function calls", "calls", calls)
				functionCalls = append(functionCalls, calls...)

				// TODO(droot): Run all function calls in parallel
				// (may have to specify in the prompt to make these function calls independent)
				for _, call := range calls {
					functionName := call.Name
					log.Info("function call", "functionName", functionName, "command", call.Arguments["command"], "modifies_resource", call.Arguments["modifies_resource"])
					a.UI.RenderOutput(ctx, fmt.Sprintf("  Running: %s\n", call.Arguments["command"]), ui.Foreground(ui.ColorGreen))
					if a.strategy.AsksForConfirmation && call.Arguments["modifies_resource"] == "no" {
						confirm := a.UI.AskForConfirmation(ctx, "  Are you sure you want to run this command (Y/n)? ")
						if !confirm {
							a.UI.RenderOutput(ctx, "Sure.\n", ui.RenderMarkdown())
							return nil
						}
					}

					output, err := a.executeAction(ctx, call, a.workDir)
					if err != nil {
						log.Error(err, "Error executing action")
						return err
					}

					currChatContent = append(currChatContent, gollm.FunctionCallResult{
						Name: functionName,
						Result: map[string]any{
							"command": call.Arguments["command"],
							"output":  output,
						},
					})

				}
			}
		}

		// If no function calls were made, we're done
		if len(functionCalls) == 0 {
			return nil
		}

		currentIteration++
	}

	// If we've reached the maximum number of iterations
	log.Info("Max iterations reached", "iterations", maxIterations)
	a.UI.RenderOutput(ctx, fmt.Sprintf("\nSorry, couldn't complete the task after %d iterations.\n", maxIterations), ui.Foreground(ui.ColorRed))
	return fmt.Errorf("max iterations reached")
}

// executeAction handles the execution of a single action
func (c *Conversation) executeAction(ctx context.Context, call gollm.FunctionCall, workDir string) (string, error) {
	log := klog.FromContext(ctx)

	tool := c.strategy.Tools.Lookup(call.Name)
	if tool == nil {
		log.Info("Unknown action: ", "action", call.Name)
		return "", fmt.Errorf("tool %q not found", call.Name)
	}

	c.recorder.Write(ctx, &journal.Event{
		Timestamp: time.Now(),
		Action:    "tool-request",
		Payload:   call.Arguments,
	})

	ctx = context.WithValue(ctx, "work_dir", workDir)
	ctx = context.WithValue(ctx, "kubeconfig", c.strategy.Kubeconfig)

	output, err := tool.Run(ctx, call.Arguments)
	if err != nil {
		return fmt.Sprintf("Error executing %q command: %v", call.Name, err), err
	}

	c.recorder.Write(ctx, &journal.Event{
		Timestamp: time.Now(),
		Action:    "tool-response",
		Payload:   output,
	})

	return output.(string), nil
}

// generateFromTemplate generates a prompt for LLM. It uses the prompt from the provides template file or default.
func (a *Strategy) generatePrompt(_ context.Context, defaultPromptTemplate string) (string, error) {
	promptTemplate := defaultPromptTemplate
	if a.PromptTemplateFile != "" {
		// Read custom template file
		content, err := os.ReadFile(a.PromptTemplateFile)
		if err != nil {
			return "", fmt.Errorf("error reading template file: %v", err)
		}
		promptTemplate = string(content)
	}

	tmpl, err := template.New("promptTemplate").Parse(promptTemplate)
	if err != nil {
		return "", fmt.Errorf("building template for prompt: %w", err)
	}

	data := map[string]string{}

	// Use a strings.Builder for efficient string concatenation
	var result strings.Builder
	// Execute the template, writing the output to the strings.Builder
	err = tmpl.Execute(&result, data)
	if err != nil {
		return "", fmt.Errorf("evaluating template for prompt: %w", err)
	}

	return result.String(), nil
}
