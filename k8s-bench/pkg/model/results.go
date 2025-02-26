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
