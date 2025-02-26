package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/GoogleCloudPlatform/kubectl-ai/k8s-bench/pkg/model"
	"sigs.k8s.io/yaml"
)

func runEvaluation(config EvalConfig) error {
	tasks, err := loadTasks(config)
	if err != nil {
		return fmt.Errorf("failed to load tasks: %w", err)
	}

	var allResults []model.TaskResult

	for taskID, task := range tasks {
		fmt.Printf("Evaluating task: %s\n", taskID)

		// Run setup if specified
		if task.Setup != "" {
			setupPath := filepath.Join(config.TasksDir, taskID, task.Setup)
			if err := runScript(setupPath, config.KubeConfig); err != nil {
				return fmt.Errorf("setup failed for task %s: %w", taskID, err)
			}
		}

		for _, llmConfig := range config.LLMConfigs {
			result := evaluateTask(config, taskID, task, llmConfig)

			if config.OutputDir != "" {
				dir := filepath.Join(config.OutputDir, taskID, llmConfig.ID)
				if err := os.MkdirAll(dir, 0755); err != nil {
					return fmt.Errorf("creating directory %q: %w", dir, err)
				}
				if err := writeToYAMLFile(filepath.Join(dir, "results.yaml"), result); err != nil {
					return fmt.Errorf("writing results to file: %w", err)
				}
			}
			allResults = append(allResults, result)
		}
	}

	printResults(allResults)
	return nil
}

// writeToYAMLFile will encode the specified object as yaml, and write it to the file.
func writeToYAMLFile(p string, obj any) error {
	data, err := yaml.Marshal(obj)
	if err != nil {
		return fmt.Errorf("marshaling to yaml: %w", err)
	}
	if err := os.WriteFile(p, data, 0644); err != nil {
		return fmt.Errorf("writing to file %q: %w", p, err)
	}
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

func evaluateTask(config EvalConfig, taskID string, task Task, llmConfig model.LLMConfig) model.TaskResult {
	result := model.TaskResult{
		Task:      taskID,
		LLMConfig: llmConfig,
	}

	// Run the agent
	cmd := exec.Command(config.AgentBin,
		"--kubeconfig", config.KubeConfig,
		"--query", task.Goal,
		"--llm-provider", llmConfig.ProviderID,
		"--model", llmConfig.ModelID,
	)

	cmd.Env = append(os.Environ(), fmt.Sprintf("KUBECONFIG=%s", config.KubeConfig))
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	fmt.Printf("\nRunning %s for task %s with %+v\n", config.AgentBin, taskID, llmConfig)

	if err := cmd.Run(); err != nil {
		result.Error = err.Error()
		return result
	}

	// Run verifier if specified
	if task.Verifier != "" {
		verifierPath := filepath.Join(config.TasksDir, taskID, task.Verifier)
		cmd = exec.Command(verifierPath)
		cmd.Env = append(os.Environ(), fmt.Sprintf("KUBECONFIG=%s", config.KubeConfig))
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		fmt.Printf("\nRunning verifier for task %s\n", taskID)

		err := cmd.Run()
		if err == nil {
			result.Result = "success"
		} else if _, ok := err.(*exec.ExitError); ok {
			// "Normal" script failure
			result.Result = "fail"
		} else {
			// Unexpected error
			result.Error = err.Error()
		}
	}

	// Run cleanup if specified
	if task.Cleanup != "" {
		cleanupPath := filepath.Join(config.TasksDir, taskID, task.Cleanup)
		if err := runScript(cleanupPath, config.KubeConfig); err != nil {
			fmt.Printf("Warning: cleanup failed for task %s: %v\n", taskID, err)
		}
	}

	return result
}

func runScript(path string, kubeconfig string) error {
	cmd := exec.Command(path)
	cmd.Env = append(os.Environ(), fmt.Sprintf("KUBECONFIG=%s", kubeconfig))
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	fmt.Printf("\nRunning script: %s\n", path)
	return cmd.Run()
}

func printResults(allResults []model.TaskResult) {
	fmt.Println("\nEvaluation Results:")
	fmt.Println("==================")

	for _, result := range allResults {
		fmt.Printf("\nTask: %s\n", result.Task)
		fmt.Printf("  LLM Config: %+vv\n", result.LLMConfig)
		fmt.Printf("    %v\n", result.Result)
		if result.Error != "" {
			fmt.Printf("    Error: %s\n", result.Error)
		}
	}
}
