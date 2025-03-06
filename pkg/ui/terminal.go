package ui

import (
	"context"
	"fmt"

	"github.com/charmbracelet/glamour"
	"k8s.io/klog/v2"
)

type TerminalUI struct {
	markdownRenderer *glamour.TermRenderer
}

var _ UI = &TerminalUI{}

func NewTerminalUI() (*TerminalUI, error) {
	mdRenderer, err := glamour.NewTermRenderer(
		glamour.WithAutoStyle(),
		glamour.WithPreservedNewLines(),
		glamour.WithEmoji(),
	)
	if err != nil {
		return nil, fmt.Errorf("error initializing the markdown renderer: %w", err)
	}
	return &TerminalUI{markdownRenderer: mdRenderer}, nil
}

func (u *TerminalUI) RenderOutput(ctx context.Context, s string, styleOptions ...StyleOption) {
	log := klog.FromContext(ctx)

	computedStyle := &style{}
	for _, opt := range styleOptions {
		opt(computedStyle)
	}

	if computedStyle.renderMarkdown {
		out, err := u.markdownRenderer.Render(s)
		if err != nil {
			log.Error(err, "Error rendering markdown")
		}
		s = out
	}

	reset := ""
	switch computedStyle.foreground {
	case ColorRed:
		fmt.Printf("\033[31m")
		reset += "\033[0m"
	case ColorGreen:
		fmt.Printf("\033[32m")
		reset += "\033[0m"
	case ColorWhite:
		fmt.Printf("\033[37m")
		reset += "\033[0m"

	case "":
	default:
		log.Info("foreground color not supported by TerminalUI", "color", computedStyle.foreground)
	}

	fmt.Printf("%s%s", s, reset)
}

func (u *TerminalUI) ClearScreen() {
	fmt.Print("\033[H\033[2J")
}
