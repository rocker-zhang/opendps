# N22 — Live Kubernetes priority handoff

N21 implemented the `JobPowerPolicy` priority path with fake Kubernetes API
tests. N22 provides the missing in-cluster controller deployment and a live
acceptance test that exercises the path through a real Kubernetes API server.
The demo remains software-only: it uses the simulator and never changes a
physical GPU power limit.

## Runtime path

The operator resolves each matching Pod's explicit
`opendps.io/gpu-indices` annotation and writes node-local assignments to
`opendps-job-boosts/resolved-assignments.json`. The controller:

1. receives its namespace and node name through the Downward API;
2. reads only the shared ConfigMap through its own ServiceAccount;
3. filters assignments to its node;
4. applies a valid new snapshot between control ticks; and
5. preserves its last-known-good snapshot during malformed data or API errors.

The controller retries in-cluster client initialization after a transient
failure. A successful resource-version change emits a
`Priority configuration applied:` log marker. This marker makes hot reload
observable without exposing workload identifiers.

Pod lifecycle reconciliation uses a silent Kopf raw-event handler. It consumes
watch event bodies without adding finalizers or writing handler progress to
Pods. A UID-keyed digest of the GPU annotation, labels, and node name suppresses
status-only updates while preserving annotation removal and deletion handling.
Each `JobPowerPolicy` also has a 30-second reconciliation timer. The timer
uses the same reconciliation path as policy create and update handlers to
republish assignments and status. This provides deterministic recovery if a
raw-event reconciliation sees a transient API failure and no later Pod event
occurs.

The demo controller has a namespaced Role granting only `get` on the
`opendps-job-boosts` ConfigMap. It cannot read other ConfigMaps or list, watch,
create, update, or delete them. The operator retains its separate
reconciliation permissions.

## Live acceptance

Build both images on a non-GB10 kind or k3s lab host:

```bash
docker build -f deploy/operator.Dockerfile -t opendps-operator:latest .
docker build -f deploy/controller.Dockerfile -t opendps-controller:latest .

# kind:
kind load docker-image opendps-operator:latest opendps-controller:latest

# k3s alternative:
docker save opendps-operator:latest opendps-controller:latest \
  | sudo k3s ctr images import -

kubectl create namespace opendps --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -f deploy/k8s/crds/
kubectl apply -f deploy/k8s/operator-deployment.yaml
kubectl apply -f deploy/k8s/controller-deployment.yaml

OPENDPS_K8S_TEST=1 .venv/bin/pytest \
  tests/test_operator_k8s_integration.py -v
```

The N22 acceptance test verifies:

- a policy matches a scheduled Pod before it has a GPU identity;
- a late GPU annotation creates the resolved assignment;
- a priority update reaches the running controller;
- the controller Pod UID and restart count do not change;
- Pod deletion clears the assignment;
- the controller ServiceAccount can get the priority ConfigMap but cannot list
  ConfigMaps or read another named ConfigMap.

The test creates only synthetic Pods and uses the controller's `--sim` mode.
It does not validate device allocation discovery, live GPU telemetry, or
physical power-cap enforcement. Automatic device identity is the N23 scope.

## Validation result

The four live acceptance tests passed on a non-GB10 x86 kind cluster. Two
independent reviewers reran the same suite and confirmed that all tests
executed without skips and that the result was not a false positive. They
checked the controller Pod continuity assertions, resource-version-bound
reload markers, empty-snapshot consumption after Pod deletion, and the
controller's restricted ConfigMap access.

This validates the Kubernetes API handoff with the software simulator. It does
not validate GPU telemetry, automatic device allocation discovery, or physical
power control.

## Public Kubernetes mechanisms

- [Downward API](https://kubernetes.io/docs/concepts/workloads/pods/downward-api/)
  supplies `metadata.namespace` and `spec.nodeName`.
- [RBAC](https://kubernetes.io/docs/reference/access-authn-authz/rbac/) keeps
  controller access scoped to its namespace.
- [kind image loading](https://kind.sigs.k8s.io/docs/user/quick-start/#loading-an-image-into-your-cluster)
  supports local image side-loading without publishing to a registry.
