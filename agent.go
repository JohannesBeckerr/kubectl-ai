package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"text/template"

	"github.com/GoogleCloudPlatform/kubectl-ai/gollm"
	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/ui"
	"k8s.io/klog/v2"
)

// LLM capability that the Agent depends on.
type LLM interface {
	GenerateContent(ctx context.Context, model, prompt string) (*ReActResponse, error)
	ListModels(ctx context.Context) ([]string, error)
}

// Agent knows how to execute a multi-step task. Goal is provided in the query argument.
type Agent struct {
	Query            string
	Model            string
	PastQueries      string
	ContentGenerator gollm.Client
	Messages         []Message
	MaxIterations    int
	CurrentIteration int
	tracePath        string
	promptFilePath   string
	Kubeconfig       string
	RemoveWorkDir    bool
	templateFile     string
}

func (a *Agent) Execute(ctx context.Context, u ui.UI) error {
	log := klog.FromContext(ctx)
	log.Info("Executing query:", "query", a.Query)

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

	WriteToFile(a.tracePath, "\n----------------------------\n")
	WriteToFile(a.promptFilePath, "\n----------------------------\n")

	// Main execution loop
	for a.CurrentIteration < a.MaxIterations {
		log.Info("Starting iteration", "iteration", a.CurrentIteration)

		// Get next action from LLM
		reActResp, err := a.AskLLM(ctx)
		if err != nil {
			log.Error(err, "Error asking LLM")
			u.RenderOutput(ctx, fmt.Sprintf("\nSorry, Couldn't complete the task. LLM error %v\n", err), ui.Foreground(ui.ColorRed))
			return err
		}

		// Log the thought process
		log.Info("Thinking...", "thought", reActResp.Thought)
		a.Trace(ctx, "assistant", fmt.Sprintf("Thought: %s", reActResp.Thought))

		// Handle final answer
		if reActResp.Answer != "" {
			log.Info("Final answer received", "answer", reActResp.Answer)
			a.Trace(ctx, "assistant", fmt.Sprintf("Final Answer: %s", reActResp.Answer))
			u.RenderOutput(ctx, reActResp.Answer, ui.RenderMarkdown())
			return nil
		}

		// Handle action
		if reActResp.Action != nil {
			log.Info("Executing action",
				"name", reActResp.Action.Name,
				"reason", reActResp.Action.Reason,
				"input", reActResp.Action.Input,
			)

			// Sanitize and prepare action
			reActResp.Action.Input = sanitizeToolInput(reActResp.Action.Input)
			a.Trace(ctx, "assistant", fmt.Sprintf("Action: Using %s tool", reActResp.Action.Name))

			// Display action details
			u.RenderOutput(ctx, fmt.Sprintf("  Running: %s", reActResp.Action.Input), ui.Foreground(ui.ColorGreen))
			u.RenderOutput(ctx, reActResp.Action.Reason, ui.RenderMarkdown())

			// Execute action
			output, err := a.executeAction(ctx, reActResp.Action, workDir)
			if err != nil {
				log.Error(err, "Error executing action")
				return err
			}

			// Record observation
			observation := fmt.Sprintf("Observation from %s:\n%s", reActResp.Action.Name, output)
			a.Trace(ctx, "system", observation)
			a.Messages = append(a.Messages, Message{Role: "system", Content: observation})
		}

		a.CurrentIteration++
	}

	// Handle max iterations reached
	log.Info("Max iterations reached", "iterations", a.CurrentIteration)
	u.RenderOutput(ctx, fmt.Sprintf("\nSorry, Couldn't complete the task after %d attempts.\n", a.MaxIterations), ui.Foreground(ui.ColorRed))
	return fmt.Errorf("max iterations reached")
}

// executeAction handles the execution of a single action
func (a *Agent) executeAction(ctx context.Context, action *Action, workDir string) (string, error) {
	log := klog.FromContext(ctx)

	switch action.Name {
	case "kubectl":
		output, err := kubectlRunner(action.Input, a.Kubeconfig, workDir)
		if err != nil {
			return fmt.Sprintf("Error executing kubectl command: %v", err), err
		}
		return output, nil
	case "cat":
		output, err := bashRunner(action.Input, workDir, a.Kubeconfig)
		if err != nil {
			return fmt.Sprintf("Error executing cat command: %v", err), err
		}
		return output, nil
	case "bash":
		output, err := bashRunner(action.Input, workDir, a.Kubeconfig)
		if err != nil {
			return fmt.Sprintf("Error executing bash command: %v", err), err
		}
		return output, nil
	default:
		a.Trace(ctx, "system", fmt.Sprintf("Error: Tool %s not found", action.Name))
		log.Info("Unknown action: ", "action", action.Name)
		return "", fmt.Errorf("unknown action: %s", action.Name)
	}
}

func (a *Agent) AskLLM(ctx context.Context) (*ReActResponse, error) {
	log := klog.FromContext(ctx)
	log.Info("Asking LLM...")

	data := Data{
		Query:       a.Query,
		PastQueries: a.PastQueries,
		History:     a.History(),
		Tools:       "kubectl, gcrane, bash",
	}

	prompt, err := a.generateFromTemplate(data)
	if err != nil {
		fmt.Println("Error generating from template:", err)
		return nil, err
	}

	log.Info("Asking LLM", "prompt", prompt)
	WriteToFile(a.promptFilePath, prompt)
	WriteToFile(a.promptFilePath, "\n========\n")

	log.Info("Thinking...", "prompt", prompt)

	response, err := a.ContentGenerator.GenerateCompletion(ctx, &gollm.CompletionRequest{
		Prompt: prompt,
	})
	if err != nil {
		return nil, fmt.Errorf("generating LLM completion: %w", err)
	}

	reActResp, err := parseReActResponse(response.Response())
	if err != nil {
		return nil, fmt.Errorf("parsing ReAct response: %w", err)
	}
	return reActResp, nil
}

func sanitizeToolInput(input string) string {
	return strings.TrimSpace(input)
}

func (a *Agent) Trace(ctx context.Context, role, content string) error {
	log := klog.FromContext(ctx)
	log.Info("Tracing...")
	msg := Message{
		Role:    role,
		Content: content,
	}
	if role != "system" {
		a.Messages = append(a.Messages, msg)
	}
	if a.tracePath == "" {
		return nil
	}
	if err := WriteToFile(a.tracePath, fmt.Sprintf("%s: %s\n", role, content)); err != nil {
		log.Error(err, "Error writing to trace file")
		// tracing is secondary to the main logic, so ignore the failures.
	}
	return nil
}

func (a *Agent) History() string {
	var history strings.Builder
	for _, msg := range a.Messages {
		history.WriteString(fmt.Sprintf("%s: %s\n", msg.Role, msg.Content))
	}
	return history.String()
}

type ReActResponse struct {
	Thought string  `json:"thought"`
	Answer  string  `json:"answer,omitempty"`
	Action  *Action `json:"action,omitempty"`
}

type Action struct {
	Name   string `json:"name"`
	Reason string `json:"reason"`
	Input  string `json:"input"`
}

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// WriteToFile appends the given content to a file.
// If the file does not exist, it is created.
// Returns an error if any operation fails.
func WriteToFile(fileName string, content string) error {
	// Open the file in append mode, create if it doesn't exist
	file, err := os.OpenFile(fileName, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return err
	}
	defer file.Close() // Ensure the file is closed after the function exits

	// Append the content to the file
	_, err = file.WriteString(content)
	if err != nil {
		return err
	}

	return nil
}

// Data represents the structure of the data to be filled into the template.
type Data struct {
	Query       string
	PastQueries string
	History     string
	Tools       string
}

// generateFromTemplate generates a string from the ReAct template using the provided data.
func (a *Agent) generateFromTemplate(data Data) (string, error) {
	var tmpl *template.Template
	var err error

	if a.templateFile != "" {
		// Read custom template file
		content, err := os.ReadFile(a.templateFile)
		if err != nil {
			return "", fmt.Errorf("error reading template file: %v", err)
		}
		tmpl, err = template.New("customTemplate").Parse(string(content))
		if err != nil {
			return "", fmt.Errorf("error parsing custom template: %v", err)
		}
	} else {
		// Use default template
		tmpl, err = template.New("reactTemplate").Parse(defaultTemplate)
		if err != nil {
			return "", err
		}
	}

	// Use a strings.Builder for efficient string concatenation
	var result strings.Builder
	// Execute the template, writing the output to the strings.Builder
	err = tmpl.Execute(&result, data)
	if err != nil {
		return "", err
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
		return nil, fmt.Errorf("parsing json %q: %w", cleaned, err)
	}
	return &reActResp, nil
}

// Move the default template to a constant
const defaultTemplate = `You are a Kubernetes Assistant tasked with answering the following query:

<query> {{.Query}} </query>

Your goal is to reason about the query and decide on the best course of action to answer it accurately.

Previous reasoning steps and observations (if any):
<previous-steps>
{{.History}}
</previous-steps>

Available tools: {{.Tools}}

Instructions:
1. Analyze the query, previous reasoning steps, and observations.
2. Decide on the next action: use a tool or provide a final answer.
3. Respond in the following JSON format:

If you need to use a tool:
{
    "thought": "Your detailed reasoning about what to do next",
    "action": {
        "name": "Tool name (kubectl, gcrane, cat, echo)",
        "reason": "Explanation of why you chose this tool (not more than 100 words)",
        "input": "complete command to be executed."
    }
}

If you have enough information to answer the query:
{
    "thought": "Your final reasoning process",
    "answer": "Your comprehensive answer to the query"
}

Remember:
- Be thorough in your reasoning.
- For creating new resources, try to create the resource using the tools available. DO NOT ask the user to create the resource.
- Prefer the tool usage that does not require any interactive input.
- Use tools when you need more information. Do not respond with the instructions on how to use the tools or what commands to run, instead just use the tool.
- Always base your reasoning on the actual observations from tool use.
- If a tool returns no results or fails, acknowledge this and consider using a different tool or approach.
- Provide a final answer only when you're confident you have sufficient information.
- If you cannot find the necessary information after using available tools, admit that you don't have enough information to answer the query confidently.
- Feel free to respond with emjois where appropriate.

Additional information from the previous queries (if any):
<previous-queries>
{{.PastQueries}}
</previous-queries>
`
