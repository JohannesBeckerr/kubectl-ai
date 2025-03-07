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

package model

type TaskResult struct {
	Task      string    `json:"name"`
	LLMConfig LLMConfig `json:"llmConfig"`
	Result    string    `json:"result"`

	// Error contains the error message, if there was an unexpected error during the execution of the test.
	// This normally indicates an infrastructure failure, rather than a test failure.
	Error string `json:"error"`
}

type LLMConfig struct {
	// ID is a short identifier for this configuration set, useful for writing logs etc
	ID string `json:"id"`

	ProviderID string `json:"provider"`
	ModelID    string `json:"model"`
	// TODO: Maybe different styles of invocation, or different temperatures etc?
}
