package main

import (
	"errors"
	"os"
	"testing"
)

func TestKubectlRunner(t *testing.T) {

	testCases := []struct {
		command             string
		expected            string
		expectedErr         error
		expectedFile        string
		expectedFileContent string
	}{
		// {
		// 	command: "kubectl get pods",
		// 	expected:    "pods",
		// 	expectedErr: nil,
		// },
		{
			command:     "kubectl edit pods",
			expected:    "interactive mode not supported for kubectl, please use non-interactive commands",
			expectedErr: errors.New("interactive mode not supported for kubectl, please use non-interactive commands"),
		},
		{

			command: `cat <<EOF > ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
 name: web-ingress
 namespace: ingress-test
spec:
 rules:
 - host: "example.com" # Replace with your desired hostname
   http:
     paths:
     - path: /app
       pathType: Prefix
       backend:
         service:
           name: web-service
           port:
             number: 80
EOF`,
			// expected:    "",
			expectedErr:  nil,
			expectedFile: "ingress.yaml",
			expectedFileContent: `apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
 name: web-ingress
 namespace: ingress-test
spec:
 rules:
 - host: "example.com" # Replace with your desired hostname
   http:
     paths:
     - path: /app
       pathType: Prefix
       backend:
         service:
           name: web-service
           port:
             number: 80
`,
		},
		{
			command: `kubectl apply -f - <<EOF
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: web-ingress
spec:
  rules:
  - host: "example.com" # Replace with your desired hostname
    http:
      paths:
      - path: /app
        pathType: Prefix
        backend:
          service:
            name: web-service
            port:
              number: 80
EOF`,
			expected:    "",
			expectedErr: nil,
		},
	}

	for _, testCase := range testCases {
		kubeconfig := ""
		workDir := ""

		output, err := kubectlRunner(testCase.command, kubeconfig, workDir)
		t.Logf("Output: %s", output)
		if err != nil {
			if testCase.expectedErr == nil {
				t.Errorf("Expected no error, but got: %v", err)
			} else if err.Error() != testCase.expectedErr.Error() {
				t.Errorf("Expected error: %v, but got: %v", testCase.expectedErr, err)
			}
		}
		if output != testCase.expected {
			t.Errorf("Expected output: %s, but got: %s", testCase.expected, output)
		}
		if testCase.expectedFile != "" {
			if _, err := os.Stat(testCase.expectedFile); os.IsNotExist(err) {
				t.Errorf("Expected file: %s, but it does not exist", testCase.expectedFile)
			}
			content, err := os.ReadFile(testCase.expectedFile)
			if err != nil {
				t.Errorf("Error reading file: %v", err)
			}
			if string(content) != testCase.expectedFileContent {
				t.Errorf("Expected file content: %s, but got: %s", testCase.expectedFileContent, string(content))
			}
		}
	}
}
