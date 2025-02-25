package main

import (
	"flag"
	"log"
	"os"
	"path/filepath"
	"strings"
)

type Task struct {
	Goal       string `yaml:"goal"`
	Setup      string `yaml:"setup,omitempty"`
	Verifier   string `yaml:"verifier,omitempty"`
	Cleanup    string `yaml:"cleanup,omitempty"`
	Difficulty string `yaml:"difficulty"`
	Disabled   bool   `yaml:"disabled,omitempty"`
}

type EvalConfig struct {
	LLMProviders []string
	Models       map[string][]string // provider -> list of models
	KubeConfig   string
	TasksDir     string
	TaskPattern  string
	AgentBin     string
}

func expandPath(path string) (string, error) {
	if strings.HasPrefix(path, "~/") {
		home, err := os.UserHomeDir()
		if err != nil {
			return "", err
		}
		path = filepath.Join(home, path[2:])
	}
	return filepath.Clean(os.ExpandEnv(path)), nil
}

func main() {
	tasksDir := flag.String("tasks-dir", "./tasks", "Directory containing evaluation tasks")
	kubeconfig := flag.String("kubeconfig", "", "Path to kubeconfig file")
	taskPattern := flag.String("task-pattern", "", "Pattern to filter tasks (e.g. 'pod' or 'redis')")
	agentBin := flag.String("agent-bin", "", "Path to kubernetes agent binary")
	llmProvider := flag.String("llm-provider", "gemini", "Specific LLM provider to evaluate (e.g. 'gemini' or 'ollama')")
	modelList := flag.String("models", "", "Comma-separated list of models to evaluate (e.g. 'gemini-1.0,gemini-2.0')")
	flag.Parse()

	if *kubeconfig == "" {
		log.Fatal("--kubeconfig is required")
	}

	expandedKubeconfig, err := expandPath(*kubeconfig)
	if err != nil {
		log.Fatalf("Failed to expand kubeconfig path: %v", err)
	}

	providers := []string{"gemini", "ollama"}
	if *llmProvider != "" {
		providers = []string{*llmProvider}
	}

	defaultModels := map[string][]string{
		"gemini": {"gemini-2.0-flash-thinking-exp-01-21"},
	}

	models := defaultModels
	if *modelList != "" {
		if *llmProvider == "" {
			log.Fatal("--llm-provider is required when --models is specified")
		}
		modelSlice := strings.Split(*modelList, ",")
		models = map[string][]string{
			*llmProvider: modelSlice,
		}
	}

	config := EvalConfig{
		LLMProviders: providers,
		Models:       models,
		KubeConfig:   expandedKubeconfig,
		TasksDir:     *tasksDir,
		TaskPattern:  *taskPattern,
		AgentBin:     *agentBin,
	}

	if err := runEvaluation(config); err != nil {
		log.Fatal(err)
	}
}
