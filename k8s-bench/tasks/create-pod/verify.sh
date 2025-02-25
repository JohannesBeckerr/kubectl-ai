#!/bin/bash
# Wait up to 30 seconds for pod to be running
for i in {1..30}; do
    if kubectl get pod web-server -n default -o jsonpath='{.status.phase}' | grep -q "Running"; then
        exit 0
    fi
    sleep 1
done

# If we get here, pod didn't reach Running state in time
exit 1 