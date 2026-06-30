# N17 — Rack-level power budget cascade

The PDN model was two effective tiers: GPU → PowerDomain → PDU. Real datacenters
add a rack budget between the PDU and the cluster — racks share a circuit, and
the sum of a rack's node budgets must fit the rack's breaker allocation. N17 adds
that tier and makes it constrain the live control loop.

## Model

- A new `Rack(name, budget_w)` dataclass and an optional `racks` section in the
  topology JSON.
- `PowerDomain` gains an optional `rack_name`. A domain with no `rack_name` is
  unaffected, so every existing topology behaves exactly as before.

## How it binds the loop

Every brain sizes GPU caps against `PDNTopology.domain_budget_w(domain)`. N17
makes that method rack-aware: when a domain belongs to a rack whose budget is
**below the sum of its domains' budgets**, the domain's budget is first scaled to
its proportional share of the rack:

```
domain_effective = domain.budget_w × rack.budget_w / Σ(rack domain budgets)
```

So the sum of GPU budgets across a rack never exceeds the rack budget — and
because *all* brains go through `domain_budget_w`, the constraint binds the live
control loop, not just a static checker. `validate_allocation()` also gains a
rack-rollup check for the static-validation path. (`from_dict` additionally now
loads `node_overhead_w` from the topology JSON.)

## Example

`deploy/topology-rackdemo.json` puts two 5000 W domains on a single 6000 W rack.
Each domain's effective budget becomes `5000 × 6000 / 10000 = 3000 W`, so the two
domains together never draw more than the 6000 W rack budget even when every GPU
is hot.

## Validation

`tests/test_rack_n17.py` covers proportional scaling (equal and unequal domains),
the no-scale case, non-membership, `validate_allocation` rack rollup, `from_dict`
parsing/back-compat, and an **end-to-end controller run** asserting the total
caps across a rack's domains stay within the rack budget through the live loop.

## Limitations

- The rack budget is a static proportional cap, not a dynamic cross-domain
  rebalancer — an idle domain's unused rack share is not lent to a hot sibling
  (that is the cross-domain-lending follow-up). The proportional split is the
  conservative, oversubscription-safe behaviour.
- Only one rack tier is modelled (PDU↔rack); a full row/datacenter cascade would
  layer additional `Rack`-like tiers.
