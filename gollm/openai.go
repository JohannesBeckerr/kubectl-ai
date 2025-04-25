package gollm

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"strings"

	"github.com/sashabaranov/go-openai"
	"k8s.io/klog/v2"
)

func init() {
	RegisterProvider("openai", openaiFactory)
}

func openaiFactory(ctx context.Context, u *url.URL) (Client, error) {
	opt := OpenAIClientOptions{}
	return NewOpenAIClient(ctx, opt)
}

// OpenAIClientOptions are the options for the OpenAI API client.
type OpenAIClientOptions struct {
	// API Key for OpenAI. Required.
	APIKey string
	// Optional custom endpoint for OpenAI API.
	Endpoint string
}

// NewOpenAIClient builds a client for the OpenAI API.
func NewOpenAIClient(ctx context.Context, opt OpenAIClientOptions) (*OpenAIClient, error) {
	apiKey := opt.APIKey
	if apiKey == "" {
		apiKey = os.Getenv("OPENAI_API_KEY")
	}
	if apiKey == "" {
		return nil, fmt.Errorf("OPENAI_API_KEY environment variable not set")
	}

	endpoint := opt.Endpoint
	if endpoint == "" {
		endpoint = os.Getenv("OPENAI_ENDPOINT")
	}

	config := openai.DefaultConfig(apiKey)
	if endpoint != "" {
		config.BaseURL = endpoint
	}

	client := openai.NewClientWithConfig(config)

	return &OpenAIClient{
		client: client,
	}, nil
}

// OpenAIClient is a client for the OpenAI API.
// It implements the Client interface.
type OpenAIClient struct {
	client *openai.Client

	// responseSchema will constrain the output to match the given schema
	responseSchema *Schema
}

var _ Client = &OpenAIClient{}

// ListModels lists the models available in the OpenAI API.
func (c *OpenAIClient) ListModels(ctx context.Context) ([]string, error) {
	models, err := c.client.ListModels(ctx)
	if err != nil {
		return nil, fmt.Errorf("error listing models: %w", err)
	}

	var modelNames []string
	for _, model := range models.Models {
		modelNames = append(modelNames, model.ID)
	}
	return modelNames, nil
}

// Close frees the resources used by the client.
func (c *OpenAIClient) Close() error {
	return nil
}

// SetResponseSchema constrains LLM responses to match the provided schema.
// Calling with nil will clear the current schema.
func (c *OpenAIClient) SetResponseSchema(schema *Schema) error {
	c.responseSchema = schema
	return nil
}

// GenerateCompletion generates a single completion for a given prompt.
func (c *OpenAIClient) GenerateCompletion(ctx context.Context, request *CompletionRequest) (CompletionResponse, error) {
	model := request.Model
	if model == "" {
		model = "gpt-4-turbo"
	}

	req := openai.ChatCompletionRequest{
		Model: model,
		Messages: []openai.ChatCompletionMessage{
			{
				Role:    openai.ChatMessageRoleUser,
				Content: request.Prompt,
			},
		},
	}

	if c.responseSchema != nil {
		// Add response format for JSON if schema is set
		req.ResponseFormat = &openai.ChatCompletionResponseFormat{
			Type: openai.ChatCompletionResponseFormatTypeJSONObject,
		}
	}

	resp, err := c.client.CreateChatCompletion(ctx, req)
	if err != nil {
		if strings.Contains(err.Error(), "status code:") {
			statusCode := http.StatusInternalServerError
			return nil, &APIError{
				StatusCode: statusCode,
				Message:    err.Error(),
				Err:        err,
			}
		}
		return nil, err
	}

	if len(resp.Choices) == 0 {
		return nil, fmt.Errorf("no choices in response")
	}

	return &OpenAICompletionResponse{
		openaiResponse: &resp,
		text:           resp.Choices[0].Message.Content,
	}, nil
}

// StartChat starts a new chat with the model.
func (c *OpenAIClient) StartChat(systemPrompt string, model string) Chat {
	if model == "" {
		model = "gpt-4-turbo"
	}

	return &OpenAIChat{
		model:  model,
		client: c.client,
		history: []openai.ChatCompletionMessage{
			{
				Role:    openai.ChatMessageRoleSystem,
				Content: systemPrompt,
			},
		},
	}
}

// OpenAIChat is a chat with the model.
// It implements the Chat interface.
type OpenAIChat struct {
	model               string
	client              *openai.Client
	history             []openai.ChatCompletionMessage
	functionDefinitions []*FunctionDefinition
}

// SetFunctionDefinitions sets the function definitions for the chat.
// This allows the LLM to call user-defined functions.
func (c *OpenAIChat) SetFunctionDefinitions(functionDefinitions []*FunctionDefinition) error {
	c.functionDefinitions = functionDefinitions
	return nil
}

// Send sends a message to the model.
// It returns a ChatResponse object containing the response from the model.
func (c *OpenAIChat) Send(ctx context.Context, contents ...any) (ChatResponse, error) {
	parts, err := c.partsToOpenAI(contents...)
	if err != nil {
		return nil, fmt.Errorf("converting parts to OpenAI format: %w", err)
	}

	// Add user message to history
	c.history = append(c.history, openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleUser,
		Content: parts,
	})

	req := openai.ChatCompletionRequest{
		Model:    c.model,
		Messages: c.history,
	}

	// Add function calling if we have function definitions
	if len(c.functionDefinitions) > 0 {
		var tools []openai.Tool
		for _, def := range c.functionDefinitions {
			schema, err := def.Parameters.ToRawSchema()
			if err != nil {
				return nil, fmt.Errorf("converting function definition schema: %w", err)
			}

			// Create a function definition with the correct type
			functionDef := &openai.FunctionDefinition{
				Name:        def.Name,
				Description: def.Description,
				Parameters:  schema,
			}

			tools = append(tools, openai.Tool{
				Type:     openai.ToolTypeFunction,
				Function: functionDef,
			})
		}
		req.Tools = tools
	}

	resp, err := c.client.CreateChatCompletion(ctx, req)
	if err != nil {
		if strings.Contains(err.Error(), "status code:") {
			statusCode := http.StatusInternalServerError
			return nil, &APIError{
				StatusCode: statusCode,
				Message:    err.Error(),
				Err:        err,
			}
		}
		return nil, err
	}

	// Add assistant response to history
	if len(resp.Choices) > 0 {
		c.history = append(c.history, resp.Choices[0].Message)
	}

	return &OpenAIChatResponse{
		openaiResponse: &resp,
	}, nil
}

// SendStreaming sends a message to the model and streams the response.
func (c *OpenAIChat) SendStreaming(ctx context.Context, contents ...any) (ChatResponseIterator, error) {
	parts, err := c.partsToOpenAI(contents...)
	if err != nil {
		return nil, fmt.Errorf("converting parts to OpenAI format: %w", err)
	}

	// Add user message to history
	c.history = append(c.history, openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleUser,
		Content: parts,
	})

	req := openai.ChatCompletionRequest{
		Model:    c.model,
		Messages: c.history,
		Stream:   true,
	}

	// Add function calling if we have function definitions
	if len(c.functionDefinitions) > 0 {
		var tools []openai.Tool
		for _, def := range c.functionDefinitions {
			schema, err := def.Parameters.ToRawSchema()
			if err != nil {
				return nil, fmt.Errorf("converting function definition schema: %w", err)
			}

			// Create a function definition with the correct type
			functionDef := &openai.FunctionDefinition{
				Name:        def.Name,
				Description: def.Description,
				Parameters:  schema,
			}

			tools = append(tools, openai.Tool{
				Type:     openai.ToolTypeFunction,
				Function: functionDef,
			})
		}
		req.Tools = tools
	}

	stream, err := c.client.CreateChatCompletionStream(ctx, req)
	if err != nil {
		if strings.Contains(err.Error(), "status code:") {
			statusCode := http.StatusInternalServerError
			return nil, &APIError{
				StatusCode: statusCode,
				Message:    err.Error(),
				Err:        err,
			}
		}
		return nil, err
	}

	// Create a channel to receive the streamed response
	responses := make(chan ChatResponse)
	errs := make(chan error, 1)

	// Start a goroutine to read from the stream
	go func() {
		defer close(responses)
		defer close(errs)
		defer stream.Close()

		var fullContent string
		var toolCalls []openai.ToolCall

		for {
			response, err := stream.Recv()
			if err != nil {
				if err.Error() == "EOF" {
					// End of stream, add the assistant message to history
					c.history = append(c.history, openai.ChatCompletionMessage{
						Role:    openai.ChatMessageRoleAssistant,
						Content: fullContent,
					})
					return
				}
				errs <- err
				return
			}

			if len(response.Choices) > 0 {
				delta := response.Choices[0].Delta

				// Accumulate content
				if delta.Content != "" {
					fullContent += delta.Content
				}

				// Accumulate tool calls
				if len(delta.ToolCalls) > 0 {
					for _, toolCall := range delta.ToolCalls {
						found := false
						for i, existingToolCall := range toolCalls {
							if existingToolCall.Index == toolCall.Index {
								// Update existing tool call
								if toolCall.Function.Name != "" {
									toolCalls[i].Function.Name = toolCall.Function.Name
								}
								if toolCall.Function.Arguments != "" {
									toolCalls[i].Function.Arguments += toolCall.Function.Arguments
								}
								found = true
								break
							}
						}
						if !found && toolCall.Index != nil && *toolCall.Index != 0 {
							// Add new tool call
							toolCalls = append(toolCalls, toolCall)
						}
					}
				}

				// Create a response for this chunk
				streamResponse := &OpenAIChatResponse{
					openaiResponse: &openai.ChatCompletionResponse{
						ID:      response.ID,
						Object:  response.Object,
						Created: response.Created,
						Model:   response.Model,
						Choices: []openai.ChatCompletionChoice{
							{
								Index: 0,
								Message: openai.ChatCompletionMessage{
									Role:      openai.ChatMessageRoleAssistant,
									Content:   fullContent,
									ToolCalls: toolCalls,
								},
							},
						},
					},
				}

				responses <- streamResponse
			}
		}
	}()

	// Return an iterator that reads from the channel
	return func(yield func(ChatResponse, error) bool) {
		for {
			select {
			case err, ok := <-errs:
				if !ok {
					return
				}
				if !yield(nil, err) {
					return
				}
			case resp, ok := <-responses:
				if !ok {
					yield(nil, nil) // Signal end of stream
					return
				}
				if !yield(resp, nil) {
					return
				}
			case <-ctx.Done():
				yield(nil, ctx.Err())
				return
			}
		}
	}, nil
}

// partsToOpenAI converts the parts to OpenAI format.
func (c *OpenAIChat) partsToOpenAI(contents ...any) (string, error) {
	if len(contents) == 0 {
		return "", nil
	}

	// For now, we just convert everything to a string
	var result string
	for _, content := range contents {
		switch v := content.(type) {
		case string:
			result += v
		case FunctionCallResult:
			// Add function result as a function message
			c.history = append(c.history, openai.ChatCompletionMessage{
				Role:       openai.ChatMessageRoleTool,
				Content:    fmt.Sprintf("%v", v.Result),
				ToolCallID: v.ID,
			})
			// Don't add this to the user message
			continue
		default:
			// Try to marshal to JSON
			b, err := json.Marshal(v)
			if err != nil {
				return "", fmt.Errorf("marshaling content to JSON: %w", err)
			}
			result += string(b)
		}
	}

	return result, nil
}

// IsRetryableError returns true if the error is retryable.
func (c *OpenAIChat) IsRetryableError(err error) bool {
	return DefaultIsRetryableError(err)
}

// OpenAIChatResponse is a response from the OpenAI API.
// It implements the ChatResponse interface.
type OpenAIChatResponse struct {
	openaiResponse *openai.ChatCompletionResponse
}

var _ ChatResponse = &OpenAIChatResponse{}

// MarshalJSON marshals the response to JSON.
func (r *OpenAIChatResponse) MarshalJSON() ([]byte, error) {
	return json.Marshal(r.openaiResponse)
}

// String returns a string representation of the response.
func (r *OpenAIChatResponse) String() string {
	if r.openaiResponse == nil || len(r.openaiResponse.Choices) == 0 {
		return ""
	}
	return r.openaiResponse.Choices[0].Message.Content
}

// UsageMetadata returns the usage metadata for the response.
func (r *OpenAIChatResponse) UsageMetadata() any {
	return r.openaiResponse.Usage
}

// Candidates returns the candidates for the response.
func (r *OpenAIChatResponse) Candidates() []Candidate {
	if r.openaiResponse == nil {
		return nil
	}

	var candidates []Candidate
	for _, choice := range r.openaiResponse.Choices {
		candidates = append(candidates, &OpenAICandidate{
			candidate: &choice,
		})
	}
	return candidates
}

// OpenAICandidate is a candidate for the response.
// It implements the Candidate interface.
type OpenAICandidate struct {
	candidate *openai.ChatCompletionChoice
}

// String returns a string representation of the response.
func (c *OpenAICandidate) String() string {
	if c.candidate == nil {
		return ""
	}
	return c.candidate.Message.Content
}

// Parts returns the parts of the candidate.
func (c *OpenAICandidate) Parts() []Part {
	if c.candidate == nil {
		return nil
	}

	var parts []Part

	// Add text content if present
	if c.candidate.Message.Content != "" {
		parts = append(parts, &OpenAIPart{
			text: c.candidate.Message.Content,
		})
	}

	// Add tool calls if present
	if len(c.candidate.Message.ToolCalls) > 0 {
		var functionCalls []FunctionCall
		for _, toolCall := range c.candidate.Message.ToolCalls {
			if toolCall.Type == openai.ToolTypeFunction {
				var args map[string]any
				if err := json.Unmarshal([]byte(toolCall.Function.Arguments), &args); err != nil {
					klog.Errorf("Error unmarshaling function arguments: %v", err)
					continue
				}

				functionCalls = append(functionCalls, FunctionCall{
					ID:        toolCall.ID,
					Name:      toolCall.Function.Name,
					Arguments: args,
				})
			}
		}

		if len(functionCalls) > 0 {
			parts = append(parts, &OpenAIPart{
				functionCalls: functionCalls,
			})
		}
	}

	return parts
}

// OpenAIPart is a part of a candidate.
// It implements the Part interface.
type OpenAIPart struct {
	text          string
	functionCalls []FunctionCall
}

// AsText returns the text of the part.
func (p *OpenAIPart) AsText() (string, bool) {
	if p.text != "" {
		return p.text, true
	}
	return "", false
}

// AsFunctionCalls returns the function calls of the part.
func (p *OpenAIPart) AsFunctionCalls() ([]FunctionCall, bool) {
	if len(p.functionCalls) > 0 {
		return p.functionCalls, true
	}
	return nil, false
}

// OpenAICompletionResponse is a response from the OpenAI API.
// It implements the CompletionResponse interface.
type OpenAICompletionResponse struct {
	openaiResponse *openai.ChatCompletionResponse
	text           string
}

var _ CompletionResponse = &OpenAICompletionResponse{}

// MarshalJSON marshals the response to JSON.
func (r *OpenAICompletionResponse) MarshalJSON() ([]byte, error) {
	return json.Marshal(r.openaiResponse)
}

// Response returns the text of the response.
func (r *OpenAICompletionResponse) Response() string {
	return r.text
}

// UsageMetadata returns the usage metadata for the response.
func (r *OpenAICompletionResponse) UsageMetadata() any {
	return r.openaiResponse.Usage
}

// String returns a string representation of the response.
func (r *OpenAICompletionResponse) String() string {
	return r.text
}
