#!/bin/bash
# Wait up to 30 seconds for pod to be running and stable
for i in {1..30}; do
    # Check if pod is running and hasn't restarted in the last check
    status=$(kubectl get pods -n debug -l app=nginx -o jsonpath='{.items[0].status.phase}')
    restarts=$(kubectl get pods -n debug -l app=nginx -o jsonpath='{.items[0].status.containerStatuses[0].restartCount}')
    
    if [[ "$status" == "Running" ]]; then
        # Wait additional 5 seconds to ensure stability
        sleep 5
        new_restarts=$(kubectl get pods -n debug -l app=nginx -o jsonpath='{.items[0].status.containerStatuses[0].restartCount}')
        if [[ "$restarts" == "$new_restarts" ]]; then
            exit 0
        fi
    fi
    sleep 1
done

# If we get here, pod didn't stabilize in time
exit 1 