# K8s-bench Evaluation Results

## Model Performance Summary

| Model | shim_disabled Success | shim_disabled Fail | shim_enabled Success | shim_enabled Fail |
|-------|------------|-----------|------------|-----------|
| gemini-2.0-flash | 10 | 1 | 7 | 3 |
| gemini-2.0-flash-thinking-exp-01-21 | 0 | 0 | 10 | 1 |
| gemma-3-27b-it | 0 | 0 | 8 | 1 |
| gemma3:12b | 0 | 0 | 5 | 2 |
| **Total** | 10 | 1 | 30 | 7 |

## Overall Summary

- Total: 53
- Success: 40 (75%)
- Fail: 8 (15%)

## Tool Use : shim_disabled

| Task | Provider | Model | Result | Error |
|------|----------|-------|--------|-------|
| configure-ingress | gemini | gemini-2.0-flash | ✅ success |  |
| create-network-policy | gemini | gemini-2.0-flash | ❌ fail |  |
| create-pod | gemini | gemini-2.0-flash | ✅ success |  |
| create-pod-mount-configmaps | gemini | gemini-2.0-flash | ✅ success |  |
| create-pod-resources-limits | gemini | gemini-2.0-flash | ✅ success |  |
| fix-crashloop | gemini | gemini-2.0-flash | ✅ success |  |
| fix-image-pull | gemini | gemini-2.0-flash | ✅ success |  |
| fix-service-routing | gemini | gemini-2.0-flash | ✅ success |  |
| list-images-for-pods | gemini | gemini-2.0-flash | ✅ success |  |
| scale-deployment | gemini | gemini-2.0-flash | ✅ success |  |
| scale-down-deployment | gemini | gemini-2.0-flash | ✅ success |  |

**shim_disabled Summary**

- Total: 11
- Success: 10 (90%)
- Fail: 1 (9%)

## Tool Use : shim_enabled

| Task | Provider | Model | Result | Error |
|------|----------|-------|--------|-------|
| configure-ingress | gemini | gemini-2.0-flash | ✅ success |  |
| configure-ingress | gemini | gemini-2.0-flash-thinking-exp-01-21 | ✅ success |  |
| configure-ingress | gemini | gemma-3-27b-it | ✅ success |  |
| create-network-policy | gemini | gemini-2.0-flash | ❌ fail |  |
| create-network-policy | gemini | gemini-2.0-flash-thinking-exp-01-21 | ❌ fail |  |
| create-network-policy | gemini | gemma-3-27b-it | ❌ fail |  |
| create-pod | gemini | gemini-2.0-flash | ✅ success |  |
| create-pod | gemini | gemini-2.0-flash-thinking-exp-01-21 | ✅ success |  |
| create-pod | gemini | gemma-3-27b-it | ✅ success |  |
| create-pod | ollama | gemma3:12b | ✅ success |  |
| create-pod-mount-configmaps | gemini | gemini-2.0-flash | ✅ success |  |
| create-pod-mount-configmaps | gemini | gemini-2.0-flash-thinking-exp-01-21 | ✅ success |  |
| create-pod-mount-configmaps | gemini | gemma-3-27b-it | ✅ success |  |
| create-pod-mount-configmaps | ollama | gemma3:12b | ❌ fail |  |
| create-pod-resources-limits | gemini | gemini-2.0-flash | ✅ success |  |
| create-pod-resources-limits | gemini | gemini-2.0-flash-thinking-exp-01-21 | ✅ success |  |
| create-pod-resources-limits | gemini | gemma-3-27b-it | ✅ success |  |
| create-pod-resources-limits | ollama | gemma3:12b | ✅ success |  |
| fix-crashloop | gemini | gemini-2.0-flash | ✅ success |  |
| fix-crashloop | gemini | gemini-2.0-flash-thinking-exp-01-21 | ✅ success |  |
| fix-crashloop | gemini | gemma-3-27b-it | ❌  | exit status 1 |
| fix-crashloop | ollama | gemma3:12b | ❌  | exit status 1 |
| fix-image-pull | gemini | gemini-2.0-flash | ✅ success |  |
| fix-image-pull | gemini | gemini-2.0-flash-thinking-exp-01-21 | ✅ success |  |
| fix-image-pull | gemini | gemma-3-27b-it | ✅ success |  |
| fix-image-pull | ollama | gemma3:12b | ❌ fail |  |
| fix-service-routing | gemini | gemini-2.0-flash | ✅ success |  |
| fix-service-routing | gemini | gemini-2.0-flash-thinking-exp-01-21 | ✅ success |  |
| fix-service-routing | gemini | gemma-3-27b-it | ❌  | exit status 1 |
| fix-service-routing | ollama | gemma3:12b | ❌  | exit status 1 |
| list-images-for-pods | ollama | gemma3:12b | ✅ success |  |
| list-images-for-pods | gemini | gemini-2.0-flash | ❌  | exit status 1 |
| list-images-for-pods | gemini | gemini-2.0-flash-thinking-exp-01-21 | ✅ success |  |
| list-images-for-pods | gemini | gemma-3-27b-it | ✅ success |  |
| scale-deployment | gemini | gemini-2.0-flash | ❌ fail |  |
| scale-deployment | gemini | gemini-2.0-flash-thinking-exp-01-21 | ✅ success |  |
| scale-deployment | gemini | gemma-3-27b-it | ✅ success |  |
| scale-deployment | ollama | gemma3:12b | ✅ success |  |
| scale-down-deployment | gemini | gemini-2.0-flash | ❌ fail |  |
| scale-down-deployment | gemini | gemini-2.0-flash-thinking-exp-01-21 | ✅ success |  |
| scale-down-deployment | gemini | gemma-3-27b-it | ✅ success |  |
| scale-down-deployment | ollama | gemma3:12b | ✅ success |  |

**shim_enabled Summary**

- Total: 42
- Success: 30 (71%)
- Fail: 7 (16%)

---

_Report generated on March 24, 2025 at 6:38 PM_
