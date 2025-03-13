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
	PreviousQueries    []string

	// Recorder captures events for diagnostics
	Recorder journal.Recorder

	RemoveWorkDir bool

	MaxIterations    int
	CurrentIteration int

	Kubeconfig          string
	AsksForConfirmation bool

	Tools tools.Tools
}

// ExecuteChatBased executes a chat-based agentic loop with the LLM using function calling.
func (a *Strategy) RunOnce(ctx context.Context, query string, previousQueries []string, u ui.UI) error {
	log := klog.FromContext(ctx)
	log.Info("Starting chat loop for query:", "query", query)

	a.PreviousQueries = previousQueries
	// Create a temporary working directory
	workDir, err := os.MkdirTemp("", "agent-workdir-*")
	if err != nil {
		log.Error(err, "Failed to create temporary working directory")
		return err
	}
	if a.RemoveWorkDir {
		defer os.RemoveAll(workDir)
	}
	log.Info("Created temporary working directory", "workDir", workDir)

	systemPrompt, err := a.generatePrompt(ctx, defaultSystemPromptChatAgent)
	if err != nil {
		log.Error(err, "Failed to generate system prompt")
		return err
	}

	// Start a new chat session
	chat := a.LLM.StartChat(systemPrompt)

	var functionDefinitions []*gollm.FunctionDefinition
	for _, tool := range a.Tools {
		functionDefinitions = append(functionDefinitions, tool.FunctionDefinition())
	}
	// Sort function definitions to help KV cache reuse
	sort.Slice(functionDefinitions, func(i, j int) bool {
		return functionDefinitions[i].Name < functionDefinitions[j].Name
	})
	if err := chat.SetFunctionDefinitions(functionDefinitions); err != nil {
		log.Error(err, "Failed to set function definitions")
		return err
	}

	// currChatContent tracks chat content that needs to be sent
	// to the LLM in each iteration of  the agentic loop below
	var currChatContent []any

	// Set the initial message to start the conversation
	currChatContent = []any{fmt.Sprintf("can you help me with query: %q", query)}

	for a.CurrentIteration < a.MaxIterations {
		log.Info("Starting iteration", "iteration", a.CurrentIteration)

		a.Recorder.Write(ctx, &journal.Event{
			Timestamp: time.Now(),
			Action:    "llm-chat",
			Payload:   []any{currChatContent},
		})

		response, err := chat.Send(ctx, currChatContent...)
		if err != nil {
			log.Error(err, "Error sending initial message")
			return err
		}

		a.Recorder.Write(ctx, &journal.Event{
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
					u.RenderOutput(ctx, textResponse, ui.RenderMarkdown())
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
					u.RenderOutput(ctx, fmt.Sprintf("  Running: %s\n", call.Arguments["command"]), ui.Foreground(ui.ColorGreen))
					if a.AsksForConfirmation && call.Arguments["modifies_resource"] == "no" {
						confirm := u.AskForConfirmation(ctx, "  Are you sure you want to run this command (Y/n)? ")
						if !confirm {
							u.RenderOutput(ctx, "Sure.\n", ui.RenderMarkdown())
							return nil
						}
					}

					output, err := a.executeAction(ctx, call, workDir)
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

		a.CurrentIteration++
	}

	// If we've reached the maximum number of iterations
	log.Info("Max iterations reached", "iterations", a.MaxIterations)
	u.RenderOutput(ctx, fmt.Sprintf("\nSorry, couldn't complete the task after %d iterations.\n", a.MaxIterations), ui.Foreground(ui.ColorRed))
	return fmt.Errorf("max iterations reached")
}

// executeAction handles the execution of a single action
func (a *Strategy) executeAction(ctx context.Context, call gollm.FunctionCall, workDir string) (string, error) {
	log := klog.FromContext(ctx)

	tool := a.Tools.Lookup(call.Name)
	if tool == nil {
		log.Info("Unknown action: ", "action", call.Name)
		return "", fmt.Errorf("tool %q not found", call.Name)
	}

	a.Recorder.Write(ctx, &journal.Event{
		Timestamp: time.Now(),
		Action:    "tool-request",
		Payload:   call.Arguments,
	})

	ctx = context.WithValue(ctx, "work_dir", workDir)
	ctx = context.WithValue(ctx, "kubeconfig", a.Kubeconfig)

	output, err := tool.Run(ctx, call.Arguments)
	if err != nil {
		return fmt.Sprintf("Error executing %q command: %v", call.Name, err), err
	}

	a.Recorder.Write(ctx, &journal.Event{
		Timestamp: time.Now(),
		Action:    "tool-response",
		Payload:   output,
	})

	return output.(string), nil
}

// generateFromTemplate generates a prompt for LLM. It uses the prompt from the provides template file or default.
func (a *Strategy) generatePrompt(_ context.Context, defaultPromptTemplate string) (string, error) {
	var tmpl *template.Template
	var err error

	promptTemplate := defaultPromptTemplate
	if a.PromptTemplateFile != "" {
		// Read custom template file
		content, err := os.ReadFile(a.PromptTemplateFile)
		if err != nil {
			return "", fmt.Errorf("error reading template file: %v", err)
		}
		promptTemplate = string(content)
	}

	tmpl, err = template.New("promptTemplate").Parse(promptTemplate)
	if err != nil {
		return "", err
	}

	data := map[string]string{}

	// Use a strings.Builder for efficient string concatenation
	var result strings.Builder
	// Execute the template, writing the output to the strings.Builder
	err = tmpl.Execute(&result, data)
	if err != nil {
		return "", err
	}

	return result.String(), nil
}
