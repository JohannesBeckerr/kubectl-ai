package llmstrategy

import (
	"context"

	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/ui"
)

type Strategy interface {
	RunOnce(ctx context.Context, userInterface ui.UI) error
}
