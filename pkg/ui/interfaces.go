package ui

import "context"

type UI interface {
	RenderOutput(ctx context.Context, s string, style ...StyleOption)

	// ClearScreen clears any output rendered to the screen
	ClearScreen()
}

type style struct {
	foreground     ColorValue
	renderMarkdown bool
}

type ColorValue string

const (
	ColorGreen ColorValue = "green"
	ColorWhite            = "white"
	ColorRed              = "red"
)

type StyleOption func(s *style)

func Foreground(color ColorValue) StyleOption {
	return func(s *style) {
		s.foreground = color
	}
}

func RenderMarkdown() StyleOption {
	return func(s *style) {
		s.renderMarkdown = true
	}
}
