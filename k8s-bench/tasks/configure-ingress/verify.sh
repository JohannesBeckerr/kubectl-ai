#!/bin/bash
# Wait for ingress to be created
if kubectl wait --for=condition=Available --timeout=30s ingress --all -n ingress-test; then
    # Check if ingress has the correct path
    if kubectl get ingress -n ingress-test -o jsonpath='{.items[*].spec.rules[*].http.paths[*].path}' | grep -q "/app"; then
        # Check if backend service is correctly configured
        if kubectl get ingress -n ingress-test -o jsonpath='{.items[*].spec.rules[*].http.paths[*].backend.service.name}' | grep -q "web-service"; then
            exit 0
        fi
    fi
fi

# If we get here, ingress wasn't configured correctly
exit 1 