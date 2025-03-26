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

package gollm

import (
	"context"
	"fmt"
	"net/url"
	"os"
)

// NewClient builds an Client based on the LLM_CLIENT env var or the provided providerID. ProviderID (if not empty) overrides the provider from LLM_CLIENT env var.
func NewClient(ctx context.Context, providerID string) (Client, error) {
	if providerID == "" {
		s := os.Getenv("LLM_CLIENT")
		if s == "" {
			return nil, fmt.Errorf("LLM_CLIENT is not set")
		}
		u, err := url.Parse(s)
		if err != nil {
			return nil, fmt.Errorf("parsing LLM_CLIENT URL: %w", err)
		}
		providerID = u.Scheme
	}

	switch providerID {
	case "gemini":
		return NewGeminiClient(ctx)
	case "vertexai":
		return NewVertexAIClient(ctx)
	case "ollama":
		return NewOllamaClient(ctx)
	case "llamacpp":
		return NewLlamaCppClient(ctx)
	default:
		return nil, fmt.Errorf("unknown LLM_CLIENT scheme %q", providerID)
	}
}
