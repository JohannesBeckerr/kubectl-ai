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

package ui

import (
	"context"
	"fmt"
	"strconv"
	"strings"

	"slices"

	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/journal"
	"github.com/charmbracelet/glamour"
	"k8s.io/klog/v2"
)

type TerminalUI struct {
	journal          journal.Recorder
	markdownRenderer *glamour.TermRenderer
}

var _ UI = &TerminalUI{}

func NewTerminalUI(journal journal.Recorder) (*TerminalUI, error) {
	mdRenderer, err := glamour.NewTermRenderer(
		glamour.WithAutoStyle(),
		glamour.WithPreservedNewLines(),
		glamour.WithEmoji(),
	)
	if err != nil {
		return nil, fmt.Errorf("error initializing the markdown renderer: %w", err)
	}
	return &TerminalUI{markdownRenderer: mdRenderer, journal: journal}, nil
}

func (u *TerminalUI) RenderOutput(ctx context.Context, s string, styleOptions ...StyleOption) {
	log := klog.FromContext(ctx)

	u.journal.Write(ctx, &journal.Event{
		Action: journal.ActionUIRender,
		Payload: map[string]any{
			"text": s,
		},
	})

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

func (u *TerminalUI) AskForConfirmation(ctx context.Context, s string, validChoices []int) int {
	log := klog.FromContext(ctx)
	fmt.Printf("%s\n", s)

	validStrs := make([]string, len(validChoices))
	for i, v := range validChoices {
		validStrs[i] = strconv.Itoa(v)
	}

	for {
		fmt.Print("  Enter your choice (number): ")
		var response string
		_, err := fmt.Scanln(&response)
		if err != nil {
			log.Error(err, "Error reading user input")
			fmt.Println("Error reading input. Please try again.")
			continue // Ask again
		}

		choice, err := strconv.Atoi(strings.TrimSpace(response))
		if err != nil {
			log.V(1).Info("Invalid input, expected a number.", "input", response)
			fmt.Println("  Invalid input. Please enter a number.")
			continue // Ask again
		}

		if slices.Contains(validChoices, choice) {
			return choice
		}

		// If not returned, the choice was invalid
		log.V(1).Info("Invalid choice entered.", "choice", choice, "validChoices", validChoices)
		fmt.Printf("  Invalid choice. Please enter one of: %s\n", strings.Join(validStrs, ", "))
		continue
	}
}
