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

package llmstrategy

import (
	"context"
	"io"

	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/ui"
)

type Strategy interface {
	// NewConversation starts a new conversation with the LLM.
	// A conversation has state that persists across rounds.
	NewConversation(ctx context.Context, userInterface ui.UI) (Conversation, error)
}

type Conversation interface {
	// Close should be called to free up resources
	io.Closer

	// RunOneRound will send the query to the LLM, and go through cycles with the LLM,
	// evaluating requested functions until we reach a stopping point.
	RunOneRound(ctx context.Context, query string) error
}
