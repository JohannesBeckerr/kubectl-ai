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

package react

import (
	"context"
	_ "embed"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"text/template"
	"time"

	"github.com/GoogleCloudPlatform/kubectl-ai/gollm"
	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/journal"
	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/llmstrategy"
	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/tools"
	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/ui"
	"k8s.io/klog/v2"
)

//go:embed react_prompt_template_default.txt
var defaultReActPromptTemplate string

type Strategy struct {
	LLM gollm.Client

	// PromptTemplateFile allows specifying a custom template file
	PromptTemplateFile string

	// Recorder captures events for diagnostics
	Recorder journal.Recorder

	MaxIterations int

	RemoveWorkDir bool

	Kubeconfig          string
	AsksForConfirmation bool

	Tools tools.Tools
}

type Conversation struct {
	strategy *Strategy
	workDir  string

	// recorder captures events for diagnostics
	recorder journal.Recorder

	userInterface ui.UI

	llmClient gollm.Client

	MaxIterations int

	previousQueries  []string
	currentIteration int
	messages         []Message
}

func (s *Strategy) NewConversation(ctx context.Context, userInterface ui.UI) (llmstrategy.Conversation, error) {
	log := klog.FromContext(ctx)

	// Create a temporary working directory
	// Create a temporary working directory
	workDir, err := os.MkdirTemp("", "agent-workdir-*")
	if err != nil {
		return nil, fmt.Errorf("creating temporary working directory: %w", err)
	}

	log.Info("Created temporary working directory", "workDir", workDir)

	return &Conversation{
		strategy:      s,
		workDir:       workDir,
		recorder:      s.Recorder,
		userInterface: userInterface,
		llmClient:     s.LLM,

		MaxIterations: s.MaxIterations,
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
	log.Info("Executing query:", "query", query)

	a.ResetHistory()

	u := a.userInterface

	// Main execution loop
	for a.currentIteration < a.MaxIterations {
		log.Info("Starting iteration", "iteration", a.currentIteration)

		// Get next action from LLM
		reActResp, err := a.askLLM(ctx, query)
		if err != nil {
			log.Error(err, "Error asking LLM")
			u.RenderOutput(ctx, fmt.Sprintf("\nSorry, Couldn't complete the task. LLM error %v\n", err), ui.Foreground(ui.ColorRed))
			return err
		}

		// Log the thought process
		log.Info("Thinking...", "thought", reActResp.Thought)
		a.addMessage(ctx, "assistant", fmt.Sprintf("Thought: %s", reActResp.Thought))

		// Handle final answer
		if reActResp.Answer != "" {
			log.Info("Final answer received", "answer", reActResp.Answer)
			a.addMessage(ctx, "assistant", fmt.Sprintf("Final Answer: %s", reActResp.Answer))
			u.RenderOutput(ctx, reActResp.Answer, ui.RenderMarkdown())
			return nil
		}

		// Handle action
		if reActResp.Action != nil {
			log.Info("Executing action",
				"name", reActResp.Action.Name,
				"reason", reActResp.Action.Reason,
				"command", reActResp.Action.Command,
				"modifies_resource", reActResp.Action.ModifiesResource,
			)

			// Sanitize and prepare action
			reActResp.Action.Command = sanitizeToolInput(reActResp.Action.Command)
			a.addMessage(ctx, "user", fmt.Sprintf("Action: %q", reActResp.Action.Command))

			// Display action details
			u.RenderOutput(ctx, fmt.Sprintf("  Running: %s", reActResp.Action.Command), ui.Foreground(ui.ColorGreen))
			u.RenderOutput(ctx, reActResp.Action.Reason, ui.RenderMarkdown())

			if a.strategy.AsksForConfirmation && reActResp.Action.ModifiesResource == "yes" {
				confirm := u.AskForConfirmation(ctx, "  Are you sure you want to run this command (Y/n)?")
				if !confirm {
					u.RenderOutput(ctx, "Sure.\n", ui.RenderMarkdown())
					return nil
				}
			}

			// Execute action
			output, err := a.executeAction(ctx, reActResp.Action, a.workDir)
			if err != nil {
				log.Error(err, "Error executing action")
				return err
			}

			// Record observation
			observation := fmt.Sprintf("Output of %q:\n%s", reActResp.Action.Command, output)
			a.addMessage(ctx, "user", observation)
		}

		a.currentIteration++
	}

	// Handle max iterations reached
	log.Info("Max iterations reached", "iterations", a.currentIteration)
	u.RenderOutput(ctx, fmt.Sprintf("\nSorry, Couldn't complete the task after %d attempts.\n", a.MaxIterations), ui.Foreground(ui.ColorRed))
	return a.recordError(ctx, fmt.Errorf("max iterations reached"))
}

// executeAction handles the execution of a single action
func (a *Conversation) executeAction(ctx context.Context, action *Action, workDir string) (string, error) {
	log := klog.FromContext(ctx)

	tool := a.strategy.Tools.Lookup(action.Name)
	if tool == nil {
		a.addMessage(ctx, "system", fmt.Sprintf("Error: Tool %s not found", action.Name))
		log.Info("Unknown action: ", "action", action.Name)
		return "", fmt.Errorf("unknown action: %s", action.Name)
	}

	ctx = context.WithValue(ctx, "kubeconfig", a.strategy.Kubeconfig)
	ctx = context.WithValue(ctx, "work_dir", workDir)

	output, err := tool.Run(ctx, map[string]any{
		"command":           action.Command,
		"modifies_resource": action.ModifiesResource,
	})
	if err != nil {
		return fmt.Sprintf("Error executing %q command: %v", action.Command, err), err
	}
	return output.(string), nil
}

// AskLLM asks the LLM for the next action, sending a prompt including the .History
func (a *Conversation) askLLM(ctx context.Context, query string) (*ReActResponse, error) {
	log := klog.FromContext(ctx)
	log.Info("Asking LLM...")

	data := PromptData{
		Query:           query,
		PreviousQueries: a.previousQueries,
		History:         a.messages,
		Tools:           strings.Join(a.strategy.Tools.Names(), ", "),
	}

	prompt, err := a.strategy.generatePrompt(ctx, defaultReActPromptTemplate, data)
	if err != nil {
		return nil, fmt.Errorf("generating prompt: %w", err)
	}

	log.Info("Thinking...", "prompt", prompt)

	response, err := a.llmClient.GenerateCompletion(ctx, &gollm.CompletionRequest{
		Prompt: prompt,
	})
	if err != nil {
		return nil, fmt.Errorf("generating LLM completion: %w", err)
	}

	a.previousQueries = append(a.previousQueries, query)

	a.recorder.Write(ctx, &journal.Event{
		Timestamp: time.Now(),
		Action:    "llm-response",
		Payload:   response,
	})

	reActResp, err := parseReActResponse(response.Response())
	if err != nil {
		return nil, fmt.Errorf("parsing ReAct response: %w", err)
	}
	return reActResp, nil
}

func sanitizeToolInput(input string) string {
	return strings.TrimSpace(input)
}

func (a *Conversation) addMessage(ctx context.Context, role, content string) error {
	log := klog.FromContext(ctx)
	log.Info("Tracing...")

	msg := Message{
		Role:    role,
		Content: content,
	}
	a.messages = append(a.messages, msg)
	a.recorder.Write(ctx, &journal.Event{
		Timestamp: time.Now(),
		Action:    "trace",
		Payload:   msg,
	})

	return nil
}

func (a *Conversation) recordError(ctx context.Context, err error) error {
	return a.recorder.Write(ctx, &journal.Event{
		Timestamp: time.Now(),
		Action:    "error",
		Payload:   err.Error(),
	})
}

func (a *Conversation) HistoryAsJSON() string {
	json, err := json.MarshalIndent(a.messages, "", "  ")
	if err != nil {
		return ""
	}
	return string(json)
}

func (a *Conversation) ResetHistory() {
	a.messages = []Message{}
}

type ReActResponse struct {
	Thought string  `json:"thought"`
	Answer  string  `json:"answer,omitempty"`
	Action  *Action `json:"action,omitempty"`
}

type Action struct {
	Name             string `json:"name"`
	Reason           string `json:"reason"`
	Command          string `json:"command"`
	ModifiesResource string `json:"modifies_resource"`
}

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// PromptData represents the structure of the data to be filled into the template.
type PromptData struct {
	Query           string
	PreviousQueries []string
	History         []Message
	Tools           string
}

func (a *PromptData) PreviousQueriesAsJSON() string {
	json, err := json.MarshalIndent(a.PreviousQueries, "", "  ")
	if err != nil {
		return ""
	}
	return string(json)
}

func (a *PromptData) HistoryAsJSON() string {
	json, err := json.MarshalIndent(a.History, "", "  ")
	if err != nil {
		return ""
	}
	return string(json)
}

// generateFromTemplate generates a prompt for LLM. It uses the prompt from the provides template file or default.
func (a *Strategy) generatePrompt(_ context.Context, defaultPromptTemplate string, data PromptData) (string, error) {
	promptTemplate := defaultPromptTemplate
	if a.PromptTemplateFile != "" {
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

	var result strings.Builder
	err = tmpl.Execute(&result, &data)
	if err != nil {
		return "", fmt.Errorf("evaluating template for prompt: %w", err)
	}
	return result.String(), nil
}

// parseReActResponse parses the LLM response into a ReActResponse struct
// This function assumes the input contains exactly one JSON code block
// formatted with ```json and ``` markers. The JSON block is expected to
// contain a valid ReActResponse object.
func parseReActResponse(input string) (*ReActResponse, error) {
	cleaned := strings.TrimSpace(input)

	const jsonBlockMarker = "```json"
	first := strings.Index(cleaned, jsonBlockMarker)
	last := strings.LastIndex(cleaned, "```")
	if first == -1 || last == -1 {
		return nil, fmt.Errorf("no JSON code block found in %q", cleaned)
	}
	cleaned = cleaned[first+len(jsonBlockMarker) : last]

	cleaned = strings.ReplaceAll(cleaned, "\n", "")
	cleaned = strings.TrimSpace(cleaned)

	var reActResp ReActResponse
	if err := json.Unmarshal([]byte(cleaned), &reActResp); err != nil {
		return nil, fmt.Errorf("parsing JSON %q: %w", cleaned, err)
	}
	return &reActResp, nil
}
