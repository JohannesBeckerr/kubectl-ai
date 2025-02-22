package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"

	"github.com/google/generative-ai-go/genai"
	"google.golang.org/api/iterator"
	"google.golang.org/api/option"
)

// Function to initialize the Gemini client.
func initGeminiClient(ctx context.Context, logger *slog.Logger, apiKey string) (*genai.Client, error) {
	logger.Info("Initializing Gemini client...")
	client, err := genai.NewClient(ctx, option.WithAPIKey(apiKey))
	if err != nil {
		logger.Error("Failed to create Gemini client", "error", err)
		return nil, err
	}
	logger.Info("Gemini client initialized successfully")
	return client, nil
}

type GeminiLLM struct {
	Model  string
	Client *genai.Client
}

func (gc *GeminiLLM) GenerateContent(ctx context.Context, model, prompt string) (*ReActResponse, error) {
	gmodel := gc.Client.GenerativeModel(model)

	resp, err := gmodel.GenerateContent(ctx, genai.Text(prompt))
	if err != nil {
		return nil, err
	}

	content := respToStr(resp)
	if content == "" {
		return nil, fmt.Errorf("empty response from LLM")
	}

	return parseReActResponse(content)
}

func (gc *GeminiLLM) ListModels(ctx context.Context) (modelNames []string, err error) {
	models := gc.Client.ListModels(ctx)

	for {
		m, err := models.Next()
		if err != nil {
			if err == iterator.Done {
				return modelNames, nil
			}
			return nil, err
		}
		modelNames = append(modelNames, strings.TrimPrefix(m.Name, "models/"))
	}
}

// parseReActResponse parses the LLM response into a ReActResponse struct
func parseReActResponse(input string) (*ReActResponse, error) {
	cleaned := strings.TrimSpace(input)

	first := strings.Index(cleaned, "```json")
	last := strings.LastIndex(cleaned, "```")
	if first == -1 || last == -1 {
		fmt.Printf("\n%s\n", cleaned)
		return nil, fmt.Errorf("no JSON code block found")
	}
	cleaned = cleaned[first+7 : last]

	cleaned = strings.ReplaceAll(cleaned, "\n", "")
	cleaned = strings.TrimSpace(cleaned)

	var reActResp ReActResponse
	if err := json.Unmarshal([]byte(cleaned), &reActResp); err != nil {
		fmt.Printf("\n%s\n", cleaned)
		return nil, err
	}
	return &reActResp, nil
}

func respToStr(resp *genai.GenerateContentResponse) string {
	for _, cand := range resp.Candidates {
		if cand.Content != nil {
			for _, part := range cand.Content.Parts {
				if _, ok := part.(genai.Text); ok {
					return fmt.Sprint(part)
				}
			}
		}
	}
	return ""
}
