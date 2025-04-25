package gollm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/url"
	"os"

	openai "github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"k8s.io/klog/v2"
)

// Register the OpenAI provider factory on package initialization.
func init() {
	if err := RegisterProvider("openai", newOpenAIClientFactory); err != nil {
		klog.Fatalf("Failed to register openai provider: %v", err)
	}
}

// newOpenAIClientFactory is the factory function for creating OpenAI clients.
func newOpenAIClientFactory(ctx context.Context, _ *url.URL) (Client, error) {
	// The URL is not currently used for OpenAI config, relies on env vars.
	return NewOpenAIClient(ctx)
}

// OpenAIClient implements the gollm.Client interface for OpenAI models.
type OpenAIClient struct {
	client openai.Client
}

// Ensure OpenAIClient implements the Client interface.
var _ Client = &OpenAIClient{}

// NewOpenAIClient creates a new client for interacting with OpenAI.
// It reads the API key and optional endpoint from environment variables
// OPENAI_API_KEY and OPENAI_ENDPOINT.
func NewOpenAIClient(ctx context.Context) (*OpenAIClient, error) {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		// The NewClient might handle this, but explicit check is safer
		return nil, errors.New("OPENAI_API_KEY environment variable not set")
	}

	endpoint := os.Getenv("OPENAI_ENDPOINT")
	if endpoint != "" {
		klog.Infof("Using custom OpenAI endpoint: %s", endpoint)
		return &OpenAIClient{
			client: openai.NewClient(option.WithBaseURL(endpoint)),
		}, nil
	}

	return &OpenAIClient{
		client: openai.NewClient(),
	}, nil
}

// Close cleans up any resources used by the client.
func (c *OpenAIClient) Close() error {
	// No specific cleanup needed for the OpenAI client currently.
	return nil
}

// StartChat starts a new chat session.
func (c *OpenAIClient) StartChat(systemPrompt, model string) Chat {
	klog.V(1).Infof("Starting new OpenAI chat session with model: %s", model)
	// Initialize history with system prompt if provided
	history := []openai.ChatCompletionMessageParamUnion{}
	if systemPrompt != "" {
		history = append(history, openai.SystemMessage(systemPrompt))
	}

	return &openAIChatSession{
		client:  c.client, // Pass the client from OpenAIClient
		history: history,
		model:   model,
		// functionDefinitions and tools will be set later via SetFunctionDefinitions
	}
}

// simpleCompletionResponse is a basic implementation of CompletionResponse.
type simpleCompletionResponse struct {
	content string
}

// Response returns the completion content.
func (r *simpleCompletionResponse) Response() string {
	return r.content
}

// UsageMetadata returns nil for now.
func (r *simpleCompletionResponse) UsageMetadata() any {
	return nil
}

// GenerateCompletion sends a completion request to the OpenAI API.
func (c *OpenAIClient) GenerateCompletion(ctx context.Context, req *CompletionRequest) (CompletionResponse, error) {
	klog.Infof("OpenAI GenerateCompletion called with model: %s", req.Model)
	klog.V(1).Infof("Prompt:\n%s", req.Prompt)

	// Use the Chat Completions API as shown in examples
	chatReq := openai.ChatCompletionNewParams{
		Model: openai.ChatModel(req.Model), // Use the model specified in the request
		Messages: []openai.ChatCompletionMessageParamUnion{
			// Assuming a simple user message structure for now
			openai.UserMessage(req.Prompt),
		},
	}

	completion, err := c.client.Chat.Completions.New(ctx, chatReq)
	if err != nil {
		return nil, fmt.Errorf("failed to generate OpenAI completion: %w", err)
	}

	// Check if there are choices and a message
	if len(completion.Choices) == 0 || completion.Choices[0].Message.Content == "" {
		return nil, errors.New("received an empty response from OpenAI")
	}

	// Return the content of the first choice
	resp := &simpleCompletionResponse{
		content: completion.Choices[0].Message.Content,
	}

	return resp, nil
}

// SetResponseSchema is not implemented yet.
func (c *OpenAIClient) SetResponseSchema(schema *Schema) error {
	klog.Warning("OpenAIClient.SetResponseSchema is not implemented yet")
	return errors.New("OpenAIClient SetResponseSchema not implemented")
}

// ListModels is not implemented yet.
func (c *OpenAIClient) ListModels(ctx context.Context) ([]string, error) {
	// TODO: Implement listing OpenAI models using c.client
	klog.Warning("OpenAIClient.ListModels is not implemented yet")
	// Return a hardcoded list for now, similar to main.go
	return []string{"gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"}, nil
}

// --- Chat Session Implementation ---

type openAIChatSession struct {
	client             openai.Client
	history            []openai.ChatCompletionMessageParamUnion
	model              string
	functionDefinitions []*FunctionDefinition // Stored in gollm format
	tools              []openai.ChatCompletionToolParam // Stored in OpenAI format
}

// Ensure openAIChatSession implements the Chat interface.
var _ Chat = (*openAIChatSession)(nil)

// SetFunctionDefinitions stores the function definitions and converts them to OpenAI format.
func (cs *openAIChatSession) SetFunctionDefinitions(defs []*FunctionDefinition) error {
	cs.functionDefinitions = defs
	cs.tools = nil // Clear previous tools
	if len(defs) > 0 {
		cs.tools = make([]openai.ChatCompletionToolParam, len(defs))
		for i, gollmDef := range defs {
			// Basic conversion, assuming schema is compatible or nil
			var params openai.FunctionParameters
			if gollmDef.Parameters != nil {
				// NOTE: This assumes gollmDef.Parameters is directly marshalable to JSON
				// that fits openai.FunctionParameters. May need refinement.
				bytes, err := gollmDef.Parameters.ToRawSchema()
				if err != nil {
					return fmt.Errorf("failed to convert schema for function %s: %w", gollmDef.Name, err)
				}
				if err := json.Unmarshal(bytes, &params); err != nil {
					return fmt.Errorf("failed to unmarshal schema for function %s: %w", gollmDef.Name, err)
				}
			}
			cs.tools[i] = openai.ChatCompletionToolParam{
				Function: openai.FunctionDefinitionParam{
					Name:        gollmDef.Name,
					Description: openai.String(gollmDef.Description),
					Parameters:  params,
				},
			}
		}
	}
	klog.V(1).Infof("Set %d function definitions for OpenAI chat session", len(cs.functionDefinitions))
	return nil
}

// Send is not fully implemented yet.
func (cs *openAIChatSession) Send(ctx context.Context, contents ...any) (ChatResponse, error) {
	// TODO: Implement actual Send logic using cs.client, cs.history, cs.model, cs.tools
	return nil, errors.New("openAIChatSession.Send not implemented")
}

// SendStreaming is not fully implemented yet.
func (cs *openAIChatSession) SendStreaming(ctx context.Context, contents ...any) (ChatResponseIterator, error) {
	// TODO: Implement actual SendStreaming logic
	return nil, errors.New("openAIChatSession.SendStreaming not implemented")
}

// IsRetryableError returns false for now.
func (cs *openAIChatSession) IsRetryableError(err error) bool {
	// TODO: Implement actual retry logic if needed
	return false
}

// --- End Chat Session Implementation ---
