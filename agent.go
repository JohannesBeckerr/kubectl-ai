package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"text/template"
	"time"

	"github.com/GoogleCloudPlatform/kubectl-ai/gollm"
	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/journal"
	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/ui"
	"k8s.io/klog/v2"
)

// Agent knows how to execute a multi-step task. Goal is provided in the query argument.
type Agent struct {
	AgentType          AgentType
	Query              string
	Model              string
	PastQueries        string
	LLM                gollm.Client
	Messages           []Message
	MaxIterations      int
	CurrentIteration   int
	Kubeconfig         string
	RemoveWorkDir      bool
	PromptTemplateFile string

	Recorder journal.Recorder
}

func (a *Agent) Execute(ctx context.Context, u ui.UI) error {
	switch a.AgentType {
	case AgentTypeChatBased:
		return a.ExecuteChatBasedLoop(ctx, u)
	case AgentTypeReAct:
		return a.ExecuteReActLoop(ctx, u)
	default:
		return fmt.Errorf("unknown agent type: %s", a.AgentType)
	}
}

func (a *Agent) ExecuteReActLoop(ctx context.Context, u ui.UI) error {
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
				"input", reActResp.Action.Input,
			)

			// Sanitize and prepare action
			reActResp.Action.Input = sanitizeToolInput(reActResp.Action.Input)
			a.addMessage(ctx, "assistant", fmt.Sprintf("Action: Using %s tool", reActResp.Action.Name))

			// Display action details
			u.RenderOutput(ctx, fmt.Sprintf("  Running: %s", reActResp.Action.Input), ui.Foreground(ui.ColorGreen))
			u.RenderOutput(ctx, reActResp.Action.Reason, ui.RenderMarkdown())

			// Execute action
			output, err := a.executeAction(ctx, reActResp.Action.Name, reActResp.Action.Input, workDir)
			if err != nil {
				log.Error(err, "Error executing action")
				return err
			}

			// Record observation
			observation := fmt.Sprintf("Observation from %s:\n%s", reActResp.Action.Name, output)
			a.addMessage(ctx, "system", observation)
		}

		a.CurrentIteration++
	}

	// Handle max iterations reached
	log.Info("Max iterations reached", "iterations", a.CurrentIteration)
	u.RenderOutput(ctx, fmt.Sprintf("\nSorry, Couldn't complete the task after %d attempts.\n", a.MaxIterations), ui.Foreground(ui.ColorRed))
	return a.recordError(ctx, fmt.Errorf("max iterations reached"))
}

// ExecuteChatBased executes a chat-based agentic loop with the LLM using function calling.
func (a *Agent) ExecuteChatBasedLoop(ctx context.Context, u ui.UI) error {
	log := klog.FromContext(ctx)
	log.Info("Starting chat loop for query:", "query", a.Query)
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

	systemPrompt, err := a.generatePrompt(ctx, defaultSystemPromptChatAgent, Data{})
	if err != nil {
		log.Error(err, "Failed to generate system prompt")
		return err
	}

	// Start a new chat session
	chat := a.LLM.StartChat(systemPrompt)

	// Define the kubectl function
	kubectlFunction := &gollm.FunctionDefinition{
		Name:        "kubectl",
		Description: "Executes kubectl command against user's Kubernetes cluster. Use this tool only when you need to query or modify the state of user's Kubernetes cluster.",
		Parameters: &gollm.Schema{
			Type: gollm.TypeObject,
			Properties: map[string]*gollm.Schema{
				"command": {
					Type: gollm.TypeString,
					Description: `The kubectl command to execute (including the kubectl prefix).

Example:
user: what pods are running in the cluster?
assistant: kubectl get pods

user: what is the status of the pod my-pod?
assistant: kubectl get pod my-pod -o jsonpath='{.status.phase}'
`,
				},
			},
		},
	}

	// make the tools available to the LLM
	if err := chat.SetFunctionDefinitions([]*gollm.FunctionDefinition{kubectlFunction}); err != nil {
		log.Error(err, "Failed to set function definitions")
		return err
	}

	// currChatContent tracks chat content that needs to be sent
	// to the LLM in each iteration of  the agentic loop below
	var currChatContent []any

	// Set the initial message to start the conversation
	currChatContent = []any{fmt.Sprintf("can you help me with query: %q", a.Query)}

	for a.CurrentIteration < a.MaxIterations {
		log.Info("Starting iteration", "iteration", a.CurrentIteration)

		response, err := chat.Send(ctx, currChatContent...)
		if err != nil {
			log.Error(err, "Error sending initial message")
			return err
		}
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
				textResponse := text
				// If we have a text response, render it
				if textResponse != "" {
					u.RenderOutput(ctx, textResponse, ui.RenderMarkdown())
					currChatContent = append(currChatContent, textResponse)
				}
			}

			// Check if it's a function call
			if calls, ok := part.AsFunctionCalls(); ok && len(calls) > 0 {
				functionCalls = append(functionCalls, calls...)

				// TODO(droot): Run all function calls in parallel
				// (may have to specify in the prompt to make these function calls independent)
				for _, call := range calls {
					functionName := call.Name
					command, _ := call.Arguments["command"].(string)

					u.RenderOutput(ctx, fmt.Sprintf("  Running: %s\n", command), ui.Foreground(ui.ColorGreen))

					output, err := a.executeAction(ctx, functionName, command, workDir)
					if err != nil {
						log.Error(err, "Error executing action")
						return err
					}

					currChatContent = append(currChatContent, gollm.FunctionCallResult{
						Name: functionName,
						Result: map[string]any{
							"command": command,
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
func (a *Agent) executeAction(ctx context.Context, actionName string, actionInput string, workDir string) (string, error) {
	log := klog.FromContext(ctx)

	switch actionName {
	case "kubectl":
		output, err := kubectlRunner(actionInput, a.Kubeconfig, workDir)
		if err != nil {
			return fmt.Sprintf("Error executing kubectl command: %v", err), err
		}
		return output, nil
	case "cat":
		output, err := bashRunner(actionInput, workDir, a.Kubeconfig)
		if err != nil {
			return fmt.Sprintf("Error executing cat command: %v", err), err
		}
		return output, nil
	case "bash":
		output, err := bashRunner(actionInput, workDir, a.Kubeconfig)
		if err != nil {
			return fmt.Sprintf("Error executing bash command: %v", err), err
		}
		return output, nil
	default:
		a.addMessage(ctx, "system", fmt.Sprintf("Error: Tool %s not found", actionName))
		log.Info("Unknown action: ", "action", actionName)
		return "", fmt.Errorf("unknown action: %s", actionName)
	}
}

// AskLLM asks the LLM for the next action, sending a prompt including the .History
func (a *Agent) AskLLM(ctx context.Context) (*ReActResponse, error) {
	log := klog.FromContext(ctx)
	log.Info("Asking LLM...")

	data := Data{
		Query:       a.Query,
		PastQueries: a.PastQueries,
		History:     a.History(),
		Tools:       "kubectl, gcrane, bash",
	}

	prompt, err := a.generatePrompt(ctx, defaultReActPromptTemplate, data)
	if err != nil {
		fmt.Println("Error generating from template:", err)
		return nil, err
	}

	log.Info("Thinking...", "prompt", prompt)

	response, err := a.LLM.GenerateCompletion(ctx, &gollm.CompletionRequest{
		Prompt: prompt,
	})
	if err != nil {
		return nil, fmt.Errorf("generating LLM completion: %w", err)
	}

	a.record(ctx, &journal.Event{
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

func (a *Agent) addMessage(ctx context.Context, role, content string) error {
	log := klog.FromContext(ctx)
	log.Info("Tracing...")

	msg := Message{
		Role:    role,
		Content: content,
	}
	a.Messages = append(a.Messages, msg)
	a.record(ctx, &journal.Event{
		Timestamp: time.Now(),
		Action:    "trace",
		Payload:   msg,
	})

	return nil
}

func (a *Agent) recordError(ctx context.Context, err error) error {
	a.record(ctx, &journal.Event{
		Timestamp: time.Now(),
		Action:    "error",
		Payload:   err.Error(),
	})
	return err
}

func (a *Agent) record(ctx context.Context, event *journal.Event) {
	log := klog.FromContext(ctx)

	log.Info("Tracing event", "event", event)

	if a.Recorder != nil {
		if err := a.Recorder.Write(ctx, event); err != nil {
			log.Error(err, "Error recording event")
		}
	}
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

// Data represents the structure of the data to be filled into the template.
type Data struct {
	Query       string
	PastQueries string
	History     string
	Tools       string
}

// generateFromTemplate generates a prompt for LLM. It uses the prompt from the provides template file or default.
func (a *Agent) generatePrompt(_ context.Context, defaultPromptTemplate string, data Data) (string, error) {
	var tmpl *template.Template
	var err error

	if a.PromptTemplateFile != "" {
		// Read custom template file
		content, err := os.ReadFile(a.PromptTemplateFile)
		if err != nil {
			return "", fmt.Errorf("error reading template file: %v", err)
		}
		tmpl, err = template.New("customTemplate").Parse(string(content))
		if err != nil {
			return "", fmt.Errorf("error parsing custom template: %v", err)
		}
	} else {
		// Use default template
		tmpl, err = template.New("promptTemplate").Parse(defaultPromptTemplate)
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

// defaultReActPromptTemplate is the default prompt template for the ReAct agent.
const defaultReActPromptTemplate = `You are a Kubernetes Assistant tasked with answering the following query:

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

// system prompt template for the chat based agent.
const defaultSystemPromptChatAgent = `You are a Kubernetes Assistant and your role is assist the user with their kubernetes related queries and tasks.
Your goal is to reason about the query and decide on the best course of action to answer it accurately.

Instructions:
- Be thorough in your reasoning.
- Don't just reason about the query, but also take actions to answer the query.
- For creating new resources, try to create the resource using the tools available. DO NOT ask the user to create the resource.
- Prefer the tool usage that does not require any interactive input.
- Use tools when you need more information.
- Always base your reasoning on the actual observations from tool use.
- If a tool returns no results or fails, acknowledge this and consider using a different tool or approach.
- Provide a final answer only when you're confident you have sufficient information.
- If you cannot find the necessary information after using available tools, admit that you don't have enough information to answer the query confidently.
- Feel free to respond with emojis where appropriate.
`
