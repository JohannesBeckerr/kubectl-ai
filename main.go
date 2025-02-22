package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"strings"
	"syscall"

	"github.com/charmbracelet/glamour"
)

// models
var geminiModels = []string{
	"gemini-2.0-flash-thinking-exp-01-21",
}

func main() {
	// non interactive execution when query is specified on the command line.
	queryFromCmd := flag.String("query", "", "query for the agent")
	maxIterations := flag.Int("max-iterations", 20, "maximum number of iterations")
	kubeconfig := flag.String("kubeconfig", "", "path to the kubeconfig file")
	llmProvider := flag.String("llm-provider", "gemini", "language model provider")
	model := flag.String("model", geminiModels[0], "language model")
	templateFile := flag.String("prompt-template-path", "", "path to custom prompt template file")
	tracePath := flag.String("trace-path", "trace.log", "path to the trace file")
	promptFilePath := flag.String("prompt-log-path", "prompt.log", "path to the prompt file")
	removeWorkDir := flag.Bool("remove-workdir", false, "remove the temporary working directory after execution")
	flag.Parse()

	logFile, err := os.OpenFile("app.log", os.O_RDWR|os.O_CREATE|os.O_APPEND, 0644)
	if err != nil {
		slog.Error("Error opening log file", "error", err)
		return
	}
	defer logFile.Close()

	textHandler := slog.NewTextHandler(logFile, nil)
	logger := slog.New(textHandler)

	mdRenderer, err := glamour.NewTermRenderer(
		glamour.WithAutoStyle(),
		glamour.WithPreservedNewLines(),
		glamour.WithEmoji(),
	)
	if err != nil {
		logger.Error("Error initializing the markdown renderer", "error", err)
		return
	}

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

	logger.Info("Application started", "pid", os.Getpid())

	go func() {
		sig := <-sigCh
		logger.Info("Received signal, shutting down...", "signal", sig)
		logger.Info("Application exiting")
		logFile.Close()
		os.Exit(0)
	}()

	ctx := context.Background()
	ctx = withLogger(ctx, logger)

	var contentGenerator LLM

	switch *llmProvider {
	case "gemini":
		apiKey := getAPIKey(logger)
		if apiKey == "" {
			fmt.Println("GEMINI_API_KEY environment variable not set")
			return // Exit if API key is not set
		}
		client, err := initGeminiClient(ctx, logger, apiKey)
		if err != nil {
			return // Exit if client initialization fails
		}
		defer client.Close()

		contentGenerator = &GeminiLLM{
			Client: client,
			Model:  *model,
		}
	default:
		logger.Error("Invalid language model provider", "provider", *llmProvider)
		return
	}

	if *queryFromCmd != "" {
		query := *queryFromCmd
		agent := Agent{
			Model:            *model,
			Query:            query,
			ContentGenerator: contentGenerator,
			MaxIterations:    *maxIterations,
			tracePath:        *tracePath,
			promptFilePath:   *promptFilePath,
			Kubeconfig:       *kubeconfig,
			RemoveWorkDir:    *removeWorkDir,
			templateFile:     *templateFile,
			markdownRenderer: mdRenderer,
		}
		agent.Execute(ctx)
		return
	}

	chatSession := session{
		Queries: []string{},
		Model:   *model,
	}

	fmt.Printf("\033[31mHey there, what can I help you with today?\033[0m")
	for {
		fmt.Printf("\n>> ")
		reader := bufio.NewReader(os.Stdin)
		query, err := reader.ReadString('\n')
		if err != nil {
			fmt.Println("Error reading input:", err)
			return
		}
		query = strings.TrimSpace(query)

		if query == "" {
			continue
		}
		switch query {
		case "reset":
			chatSession.Queries = []string{}
			clearScreen()
		case "clear":
			clearScreen()
		case "exit", "quit":
			fmt.Println("Allright...bye.")
			return
		case "models":
			modelNames, err := contentGenerator.ListModels(ctx)
			if err != nil {
				fmt.Println("Error listing models:", err)
				continue
			}
			fmt.Println("Available models:")
			for _, modelName := range modelNames {
				fmt.Println(modelName)
			}
		default:
			if strings.HasPrefix(query, "model") {
				parts := strings.Split(query, " ")
				if len(parts) > 2 {
					fmt.Println("Invalid model command. expected format: model <model-name>")
					continue
				}
				if len(parts) == 1 {
					out := fmt.Sprintf("Current model is `%s`\n", chatSession.Model)
					rendered, _ := mdRenderer.Render(out)
					fmt.Println(rendered)
					continue
				}
				chatSession.Model = parts[1]
				fmt.Printf("Model set to `%s`\n", chatSession.Model)
				continue
			}
			agent := Agent{
				Model:            chatSession.Model,
				Query:            query,
				PastQueries:      chatSession.PreviousQueries(),
				ContentGenerator: contentGenerator,
				MaxIterations:    *maxIterations,
				tracePath:        *tracePath,
				promptFilePath:   *promptFilePath,
				Kubeconfig:       *kubeconfig,
				RemoveWorkDir:    *removeWorkDir,
				templateFile:     *templateFile,
				markdownRenderer: mdRenderer,
			}
			agent.Execute(ctx)
			chatSession.Queries = append(chatSession.Queries, query)
		}
	}
}

// session represents each the chat session.
type session struct {
	Queries []string `json:"queries"`
	Model   string   `json:"model"`
}

func (s *session) PreviousQueries() string {
	return strings.Join(s.Queries, "\n")
}

// Logger key for context
type loggerKey struct{}

// Function to create a new context with the logger.
func withLogger(ctx context.Context, logger *slog.Logger) context.Context {
	return context.WithValue(ctx, loggerKey{}, logger)
}

// Function to extract the logger from the context.
func loggerFromContext(ctx context.Context) *slog.Logger {
	logger, ok := ctx.Value(loggerKey{}).(*slog.Logger)
	if !ok {
		return slog.Default()
	}
	return logger
}

// Function to get the Gemini API key from environment variable.
func getAPIKey(logger *slog.Logger) string {
	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		logger.Error("GEMINI_API_KEY environment variable not set")
		return ""
	}
	return apiKey
}

func clearScreen() {
	fmt.Print("\033[H\033[2J")
}
