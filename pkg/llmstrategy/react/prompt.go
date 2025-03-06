package react

import (
	"fmt"
	"html/template"
	"os"
	"strings"
)

// generateFromTemplate generates a string from the ReAct template using the provided data.
func (a *Strategy) generateFromTemplate(data Data) (string, error) {
	var tmpl *template.Template
	var err error

	if a.TemplateFile != "" {
		// Read custom template file
		content, err := os.ReadFile(a.TemplateFile)
		if err != nil {
			return "", fmt.Errorf("error reading template file: %v", err)
		}
		tmpl, err = template.New("customTemplate").Parse(string(content))
		if err != nil {
			return "", fmt.Errorf("error parsing custom template: %v", err)
		}
	} else {
		// Use default template
		tmpl, err = template.New("reactTemplate").Parse(defaultTemplate)
		if err != nil {
			return "", err
		}
	}

	// Use a strings.Builder for efficient string concatenation
	var result strings.Builder
	// Execute the template, writing the output to the strings.Builder
	err = tmpl.Execute(&result, data)
	if err != nil {
		return "", err
	}

	return result.String(), nil
}

// Move the default template to a constant
const defaultTemplate = `You are a Kubernetes Assistant tasked with answering the following query:

<query> {{.Query}} </query>

Your goal is to reason about the query and decide on the best course of action to answer it accurately.

Previous reasoning steps and observations (if any):
<previous-steps>
{{.History}}
</previous-steps>

Available tools: {{.Tools}}

Instructions:
1. Analyze the query, previous reasoning steps, and observations.
2. Decide on the next action: use a tool or provide a final answer.
3. Respond in the following JSON format:

If you need to use a tool:
{
    "thought": "Your detailed reasoning about what to do next",
    "action": {
        "name": "Tool name (kubectl, gcrane, cat, echo)",
        "reason": "Explanation of why you chose this tool (not more than 100 words)",
        "input": "complete command to be executed."
    }
}

If you have enough information to answer the query:
{
    "thought": "Your final reasoning process",
    "answer": "Your comprehensive answer to the query"
}

Remember:
- Be thorough in your reasoning.
- For creating new resources, try to create the resource using the tools available. DO NOT ask the user to create the resource.
- Prefer the tool usage that does not require any interactive input.
- Use tools when you need more information. Do not respond with the instructions on how to use the tools or what commands to run, instead just use the tool.
- Always base your reasoning on the actual observations from tool use.
- If a tool returns no results or fails, acknowledge this and consider using a different tool or approach.
- Provide a final answer only when you're confident you have sufficient information.
- If you cannot find the necessary information after using available tools, admit that you don't have enough information to answer the query confidently.
- Feel free to respond with emjois where appropriate.

Additional information from the previous queries (if any):
<previous-queries>
{{.PastQueries}}
</previous-queries>
`
