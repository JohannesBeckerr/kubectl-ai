#!/bin/bash
# Wait up to 30 seconds for ingress to be created and configured
for i in {1..30}; do
    # Check if ingress exists and has the correct path
    if kubectl get ingress -n ingress-test -o jsonpath='{.items[*].spec.rules[*].http.paths[*].path}' | grep -q "/app"; then
        # Check if backend service is correctly configured
        if kubectl get ingress -n ingress-test -o jsonpath='{.items[*].spec.rules[*].http.paths[*].backend.service.name}' | grep -q "web-service"; then
            exit 0
        fi
    fi
    sleep 1
done

# If we get here, ingress wasn't configured correctly
exit 1 