package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"sigs.k8s.io/yaml"
)

func runEvaluation(config EvalConfig) error {
	tasks, err := loadTasks(config)
	if err != nil {
		return fmt.Errorf("failed to load tasks: %w", err)
	}

	results := make(map[string]map[string]map[string]bool) // task -> provider -> model -> success

	for taskID, task := range tasks {
		fmt.Printf("Evaluating task: %s\n", taskID)

		// Run setup if specified
		if task.Setup != "" {
			setupPath := filepath.Join(config.TasksDir, taskID, task.Setup)
			if err := runScript(setupPath, config.KubeConfig); err != nil {
				return fmt.Errorf("setup failed for task %s: %w", taskID, err)
			}
		}

		for _, provider := range config.LLMProviders {
			results[taskID] = make(map[string]map[string]bool)
			results[taskID][provider] = make(map[string]bool)

			for _, model := range config.Models[provider] {
				success, err := evaluateTask(config, taskID, task, provider, model)
				if err != nil {
					fmt.Printf("Error evaluating task %s with %s/%s: %v\n", taskID, provider, model, err)
					results[taskID][provider][model] = false
					continue
				}
				results[taskID][provider][model] = success
			}
		}
	}

	printResults(results)
	return nil
}

func loadTasks(config EvalConfig) (map[string]Task, error) {
	tasks := make(map[string]Task)

	entries, err := os.ReadDir(config.TasksDir)
	if err != nil {
		return nil, err
	}

	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}

		taskID := entry.Name()
		if config.TaskPattern != "" && !strings.Contains(taskID, config.TaskPattern) {
			continue
		}

		taskFile := filepath.Join(config.TasksDir, taskID, "task.yaml")

		data, err := os.ReadFile(taskFile)
		if err != nil {
			return nil, fmt.Errorf("failed to read task file %s: %w", taskFile, err)
		}

		var task Task
		if err := yaml.Unmarshal(data, &task); err != nil {
			return nil, fmt.Errorf("failed to parse task file %s: %w", taskFile, err)
		}

		// Skip disabled tasks
		if task.Disabled {
			fmt.Printf("Skipping disabled task: %s\n", taskID)
			continue
		}

		tasks[taskID] = task
	}

	return tasks, nil
}

func evaluateTask(config EvalConfig, taskID string, task Task, provider, model string) (bool, error) {
	// Run kuba
	cmd := exec.Command(config.AgentBin,
		"--kubeconfig", config.KubeConfig,
		"--query", task.Goal,
		"--llm-provider", provider,
		"--model", model,
	)

	cmd.Env = append(os.Environ(), fmt.Sprintf("KUBECONFIG=%s", config.KubeConfig))
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	fmt.Printf("\nRunning kuba for task %s with %s/%s\n", taskID, provider, model)

	if err := cmd.Run(); err != nil {
		return false, fmt.Errorf("kuba failed: %w", err)
	}

	success := true
	// Run verifier if specified
	if task.Verifier != "" {
		verifierPath := filepath.Join(config.TasksDir, taskID, task.Verifier)
		cmd = exec.Command(verifierPath)
		cmd.Env = append(os.Environ(), fmt.Sprintf("KUBECONFIG=%s", config.KubeConfig))
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		fmt.Printf("\nRunning verifier for task %s\n", taskID)

		err := cmd.Run()
		success = err == nil
		if !success {
			fmt.Printf("Verifier failed with error: %v\n", err)
		}
	}

	// Run cleanup if specified
	if task.Cleanup != "" {
		cleanupPath := filepath.Join(config.TasksDir, taskID, task.Cleanup)
		if err := runScript(cleanupPath, config.KubeConfig); err != nil {
			fmt.Printf("Warning: cleanup failed for task %s: %v\n", taskID, err)
		}
	}

	return success, nil
}

func runScript(path string, kubeconfig string) error {
	cmd := exec.Command(path)
	cmd.Env = append(os.Environ(), fmt.Sprintf("KUBECONFIG=%s", kubeconfig))
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	fmt.Printf("\nRunning script: %s\n", path)
	return cmd.Run()
}

func printResults(results map[string]map[string]map[string]bool) {
	fmt.Println("\nEvaluation Results:")
	fmt.Println("==================")

	for taskID, providerResults := range results {
		fmt.Printf("\nTask: %s\n", taskID)
		for provider, modelResults := range providerResults {
			fmt.Printf("  Provider: %s\n", provider)
			for model, success := range modelResults {
				fmt.Printf("    %s: %v\n", model, success)
			}
		}
	}
}
