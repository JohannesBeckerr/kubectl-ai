// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"

	"github.com/GoogleCloudPlatform/kubectl-ai/gollm"
	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/agent"
	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/journal"
	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/tools"
	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/ui"

	"k8s.io/klog/v2"
	"sigs.k8s.io/yaml"
)

// Version will be set at build time
var Version = "0.1.0-dev"

// models
var geminiModels = []string{
	"gemini-2.5-pro-preview-03-25",
	"gemini-2.0-flash",
}

func main() {
	ctx := context.Background()

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		sig := <-sigCh
		fmt.Fprintf(os.Stderr, "Received signal, shutting down... %s\n", sig)
		klog.Flush()
		os.Exit(0)
	}()

	if err := run(ctx); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

type Options struct {
	ProviderID string `json:"llmProvider,omitempty"`
	ModelID    string `json:"model,omitempty"`

	// SkipPermissions is a flag to skip asking for confirmation before executing kubectl commands
	// that modifies resources in the cluster.
	SkipPermissions bool `json:"skipPermissions,omitempty"`

	// EnableToolUseShim is a flag to enable tool use shim.
	// TODO(droot): figure out a better way to discover if the model supports tool use
	// and set this automatically.
	EnableToolUseShim bool `json:"enableToolUseShim,omitempty"`

	// Quiet flag indicates if the agent should run in non-interactive mode.
	// It requires a query to be provided as a positional argument.
	Quiet bool `json:"quiet,omitempty"`

	MCPServer bool
}

func (o *Options) InitDefaults() {
	o.ProviderID = "gemini://"
	o.ModelID = geminiModels[0]
	// by default, confirm before executing kubectl commands that modify resources in the cluster.
	o.SkipPermissions = false
	o.MCPServer = false

	// We now default to our strongest model (gemini-2.5-pro-exp-03-25) which supports tool use natively.
	// so we don't need shim.
	o.EnableToolUseShim = false
}

func (o *Options) LoadConfiguration(b []byte) error {
	if err := yaml.Unmarshal(b, &o); err != nil {
		return fmt.Errorf("parsing configuration: %w", err)
	}
	return nil
}

func (o *Options) LoadConfigurationFile() error {
	configPaths := []string{
		"{CONFIG}/kubectl-ai/config.yaml",
		"{HOME}/.config/kubectl-ai/config.yaml",
	}

	for _, configPath := range configPaths {
		// Try to load configuration
		tokens := strings.Split(configPath, "/")
		for i, token := range tokens {
			if token == "{CONFIG}" {
				configDir, err := os.UserConfigDir()
				if err != nil {
					return fmt.Errorf("getting user config directory: %w", err)
				}
				tokens[i] = configDir
			}
			if token == "{HOME}" {
				homeDir, err := os.UserHomeDir()
				if err != nil {
					return fmt.Errorf("getting user home directory: %w", err)
				}
				tokens[i] = homeDir
			}
		}
		configPath = filepath.Join(tokens...)
		configBytes, err := os.ReadFile(configPath)
		if err != nil {
			if os.IsNotExist(err) {
				// ignore
			} else {
				fmt.Fprintf(os.Stderr, "warning: could not load defaults from %q: %v\n", configPath, err)
			}
		}
		if len(configBytes) > 0 {
			if err := o.LoadConfiguration(configBytes); err != nil {
				fmt.Fprintf(os.Stderr, "warning: error loading configuration from %q: %v\n", configPath, err)
			}
		}
	}

	return nil
}

func run(ctx context.Context) error {
	// Command line flags
	var opt Options
	opt.InitDefaults()

	if err := opt.LoadConfigurationFile(); err != nil {
		return fmt.Errorf("loading configuration file: %w", err)
	}

	maxIterations := flag.Int("max-iterations", 20, "maximum number of iterations agent will try before giving up")
	kubeconfig := flag.String("kubeconfig", "", "path to the kubeconfig file")
	promptTemplateFile := flag.String("prompt-template-file", "", "path to custom prompt template file")
	tracePath := flag.String("trace-path", "trace.log", "path to the trace file")
	removeWorkDir := flag.Bool("remove-workdir", false, "remove the temporary working directory after execution")

	flag.StringVar(&opt.ProviderID, "llm-provider", opt.ProviderID, "language model provider")
	flag.StringVar(&opt.ModelID, "model", opt.ModelID, "language model e.g. gemini-2.0-flash-thinking-exp-01-21, gemini-2.0-flash")
	flag.BoolVar(&opt.SkipPermissions, "skip-permissions", opt.SkipPermissions, "(dangerous) skip asking for confirmation before executing kubectl commands that modify resources")
	flag.BoolVar(&opt.MCPServer, "mcp-server", opt.MCPServer, "run in MCP server mode")
	flag.BoolVar(&opt.EnableToolUseShim, "enable-tool-use-shim", opt.EnableToolUseShim, "enable tool use shim")
	flag.BoolVar(&opt.Quiet, "quiet", opt.Quiet, "run in non-interactive mode, requires a query to be provided as a positional argument")

	// add commandline flags for logging
	klog.InitFlags(nil)

	flag.Set("logtostderr", "false") // disable logging to stderr
	flag.Set("log_file", "/tmp/kubectl-ai.log")

	flag.Parse()

	defer klog.Flush()

	// Handle kubeconfig with priority: command-line arg > env var > default path
	kubeconfigPath := *kubeconfig
	if kubeconfigPath == "" {
		// Check environment variable
		kubeconfigPath = os.Getenv("KUBECONFIG")
		if kubeconfigPath == "" {
			// Use default path
			homeDir, err := os.UserHomeDir()
			if err != nil {
				return fmt.Errorf("error getting user home directory: %w", err)
			}
			kubeconfigPath = filepath.Join(homeDir, ".kube", "config")
		}
	}
	if opt.MCPServer {
		workDir := "/tmp/kubectl-ai-mcp"
		if err := os.MkdirAll(workDir, 0755); err != nil {
			return fmt.Errorf("error creating work directory: %w", err)
		}
		mcpServer, err := newKubectlMCPServer(ctx, kubeconfigPath, tools.Default(), workDir)
		if err != nil {
			return fmt.Errorf("creating mcp server: %w", err)
		}
		return mcpServer.Serve(ctx)
	}

	// Check for positional arguments (after all flags are parsed)
	args := flag.Args()
	var queryFromCmd string

	// Check if stdin has data (is not a terminal)
	stdinInfo, _ := os.Stdin.Stat()
	stdinHasData := (stdinInfo.Mode() & os.ModeCharDevice) == 0

	// Handle positional arguments and stdin
	if len(args) > 1 {
		return fmt.Errorf("only one positional argument (query) is allowed")
	} else if stdinHasData {
		// Read from stdin
		scanner := bufio.NewScanner(os.Stdin)
		var queryBuilder strings.Builder

		// If we have a positional argument, use it as a prefix
		if len(args) == 1 {
			queryBuilder.WriteString(args[0])
			queryBuilder.WriteString("\n")
		}

		// Read the rest from stdin
		for scanner.Scan() {
			queryBuilder.WriteString(scanner.Text())
			queryBuilder.WriteString("\n")
		}
		if err := scanner.Err(); err != nil {
			return fmt.Errorf("error reading from stdin: %w", err)
		}

		queryFromCmd = strings.TrimSpace(queryBuilder.String())
		if queryFromCmd == "" {
			return fmt.Errorf("no query provided from stdin")
		}
	} else if len(args) == 1 {
		// Just use the positional argument as the query
		queryFromCmd = args[0]
	}

	klog.Info("Application started", "pid", os.Getpid())

	llmClient, err := gollm.NewClient(ctx, opt.ProviderID)
	if err != nil {
		return fmt.Errorf("creating llm client: %w", err)
	}
	defer llmClient.Close()

	var recorder journal.Recorder
	if *tracePath != "" {
		fileRecorder, err := journal.NewFileRecorder(*tracePath)
		if err != nil {
			return fmt.Errorf("creating trace recorder: %w", err)
		}
		defer fileRecorder.Close()
		recorder = fileRecorder
	} else {
		// Ensure we always have a recorder, to avoid nil checks
		recorder = &journal.LogRecorder{}
		defer recorder.Close()
	}

	doc := ui.NewDocument()

	// since stdin is already consumed, we use TTY for taking input from user
	useTTYForInput := stdinHasData
	u, err := ui.NewTerminalUI(doc, recorder, useTTYForInput)
	if err != nil {
		return err
	}

	conversation := &agent.Conversation{
		Model:              opt.ModelID,
		Kubeconfig:         kubeconfigPath,
		LLM:                llmClient,
		MaxIterations:      *maxIterations,
		PromptTemplateFile: *promptTemplateFile,
		Tools:              tools.Default(),
		Recorder:           recorder,
		RemoveWorkDir:      *removeWorkDir,
		SkipPermissions:    opt.SkipPermissions,
		EnableToolUseShim:  opt.EnableToolUseShim,
	}

	err = conversation.Init(ctx, doc)
	if err != nil {
		return fmt.Errorf("starting conversation: %w", err)
	}
	defer conversation.Close()

	chatSession := session{
		model:        opt.ModelID,
		doc:          doc,
		ui:           u,
		conversation: conversation,
		LLM:          llmClient,
	}

	if opt.Quiet {
		if queryFromCmd == "" {
			return fmt.Errorf("quiet mode requires a query to be provided as a positional argument")
		}
		return chatSession.answerQuery(ctx, queryFromCmd)
	}

	return chatSession.repl(ctx, queryFromCmd)
}

// session represents the user chat session (interactive/non-interactive both)
type session struct {
	model           string
	ui              ui.UI
	doc             *ui.Document
	conversation    *agent.Conversation
	availableModels []string
	LLM             gollm.Client
}

// repl is a read-eval-print loop for the chat session.
func (s *session) repl(ctx context.Context, initialQuery string) error {
	query := initialQuery
	if query == "" {
		s.doc.AddBlock(ui.NewAgentTextBlock().SetText("Hey there, what can I help you with today?"))
	}
	for {
		if query == "" {
			input := ui.NewInputTextBlock()
			s.doc.AddBlock(input)

			userInput, err := input.Observable().Wait()
			if err != nil {
				if err == io.EOF {
					// Use hit control-D, or was piping and we reached the end of stdin.
					// Not a "big" problem
					return nil
				}
				return fmt.Errorf("reading input: %w", err)
			}
			query = strings.TrimSpace(userInput)
		}

		switch {
		case query == "":
			continue
		case query == "reset":
			err := s.conversation.Init(ctx, s.doc)
			if err != nil {
				return err
			}
		case query == "clear":
			s.ui.ClearScreen()
		case query == "exit" || query == "quit":
			// s.ui.RenderOutput(ctx, "Allright...bye.\n")
			return nil
		default:
			if err := s.answerQuery(ctx, query); err != nil {
				errorBlock := &ui.ErrorBlock{}
				errorBlock.SetText(fmt.Sprintf("Error: %v\n", err))
				s.doc.AddBlock(errorBlock)
			}
		}
		// Reset query to empty string so that we prompt for input again
		query = ""
	}
}

func (s *session) listModels(ctx context.Context) ([]string, error) {
	if s.availableModels == nil {
		modelNames, err := s.LLM.ListModels(ctx)
		if err != nil {
			return nil, fmt.Errorf("listing models: %w", err)
		}
		s.availableModels = modelNames
	}
	return s.availableModels, nil
}

func (s *session) answerQuery(ctx context.Context, query string) error {
	switch {
	case query == "model":
		infoBlock := &ui.AgentTextBlock{}
		infoBlock.AppendText(fmt.Sprintf("Current model is `%s`\n", s.model))
		s.doc.AddBlock(infoBlock)

	case query == "version":
		infoBlock := &ui.AgentTextBlock{}
		infoBlock.AppendText(fmt.Sprintf("Client version: `%s`\n", Version))
		s.doc.AddBlock(infoBlock)

	case query == "models":
		models, err := s.listModels(ctx)
		if err != nil {
			return fmt.Errorf("listing models: %w", err)
		}
		infoBlock := &ui.AgentTextBlock{}
		infoBlock.AppendText("\n  Available models:\n")
		infoBlock.AppendText(strings.Join(models, "\n"))
		s.doc.AddBlock(infoBlock)

	default:
		return s.conversation.RunOneRound(ctx, query)
	}
	return nil
}
