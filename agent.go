package main

import (
	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/journal"
	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/llmstrategy"
)

// Agent knows how to execute a multi-step task. Goal is provided in the query argument.
type Agent struct {
	Model string

	Recorder journal.Recorder

	Strategy llmstrategy.Strategy
}
