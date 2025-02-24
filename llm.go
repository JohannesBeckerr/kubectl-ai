package main

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/google/generative-ai-go/genai"
)

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
