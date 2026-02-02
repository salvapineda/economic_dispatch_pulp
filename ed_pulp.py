"""
Economic Dispatch solver using PuLP with CBC solver.

This solver takes a fixed unit commitment schedule (from Julia's MILP solution)
and optimizes the power dispatch to minimize operational costs.

The LP solution differs slightly from Julia's MILP solution because:
- Julia solves UC+ED jointly as a MILP
- This solver only optimizes ED with fixed commitment as an LP

Expected difference: ~0.16% (LP dispatch vs MILP dispatch)
"""
import gzip
import json
from pathlib import Path
import numpy as np
import pulp as pl


def solve_economic_dispatch(instance_name):
    """
    Solve economic dispatch for a given instance with fixed commitment.
    
    Args:
        instance_name: Name of the instance (e.g., 'instance_2027_Q3_59')
    
    Returns:
        Dictionary with solution variables
    """
    # Load input data
    input_path = Path(instance_name) / "InputData.json.gz"
    with gzip.open(input_path, 'rt') as f:
        input_data = json.load(f)
    
    # Load output data (commitment from Julia)
    output_path = Path(instance_name) / "OutputData.json.gz"
    with gzip.open(output_path, 'rt') as f:
        output_data = json.load(f)
    
    # Parameters
    T = input_data['Parameters']['Time horizon (h)']
    penalty = input_data['Parameters']['Power balance penalty ($/MW)']
    
    # Create optimization model
    model = pl.LpProblem("Economic_Dispatch", pl.LpMinimize)
    
    # === VARIABLES ===
    
    # Thermal generator variables
    p_above = {}  # Power above minimum
    ps = {}  # Piecewise segment power
    
    for g in input_data['Generators']:
        if input_data['Generators'][g]['Type'] == 'Thermal':
            curve_power = input_data['Generators'][g]['Production cost curve (MW)']
            pmin, pmax = curve_power[0], curve_power[-1]
            segments_power = np.diff(curve_power)
            
            for t in range(T):
                commit = output_data['Is on'][g][t]
                
                # Power above minimum
                p_above[g, t] = pl.LpVariable(f"p_above_{g}_{t}", 0, (pmax - pmin) * commit)
                
                # Segment variables (NOT scaled by commit, just like model.py line 176)
                for s, seg_power in enumerate(segments_power):
                    ps[g, t, s] = pl.LpVariable(f"ps_{g}_{t}_{s}", 0, seg_power)
    
    # Profiled generators (wind, solar, hydro) - some have time-varying capacity, some don't
    profiled = {}
    for g in input_data['Generators']:
        if input_data['Generators'][g]['Type'] == 'Profiled':
            max_power_data = input_data['Generators'][g]['Maximum power (MW)']
            for t in range(T):
                # Handle both time-series (wind/solar) and scalar (hydro) max power
                if isinstance(max_power_data, list):
                    max_power = max_power_data[t]
                else:
                    max_power = max_power_data
                profiled[g, t] = pl.LpVariable(f"profiled_{g}_{t}", 0, max_power)
    
    # Storage variables
    charge = {}
    discharge = {}
    level = {}
    
    for s in input_data['Storage units']:
        max_charge = input_data['Storage units'][s]['Maximum charge rate (MW)']
        max_discharge = input_data['Storage units'][s]['Maximum discharge rate (MW)']
        max_level = input_data['Storage units'][s]['Maximum level (MWh)']
        init_level = input_data['Storage units'][s]['Initial level (MWh)']
        
        for t in range(T):
            charge[s, t] = pl.LpVariable(f"charge_{s}_{t}", 0, max_charge)
            discharge[s, t] = pl.LpVariable(f"discharge_{s}_{t}", 0, max_discharge)
            
            # Final period must return to initial level
            if t < T - 1:
                level[s, t] = pl.LpVariable(f"level_{s}_{t}", 0, max_level)
            else:
                level[s, t] = pl.LpVariable(f"level_{s}_{t}", init_level, max_level)
    
    # Load shedding
    load_shed = {}
    for t in range(T):
        demand = input_data['Buses']['b1']['Load (MW)'][t]
        load_shed[t] = pl.LpVariable(f"load_shed_{t}", 0, demand)
    
    # === OBJECTIVE FUNCTION ===
    
    obj = 0
    
    # Piecewise linear production cost
    for g in input_data['Generators']:
        if input_data['Generators'][g]['Type'] == 'Thermal':
            curve_power = input_data['Generators'][g]['Production cost curve (MW)']
            curve_cost = input_data['Generators'][g]['Production cost curve ($)']
            marginal_costs = np.diff(curve_cost) / np.diff(curve_power)
            
            for t in range(T):
                for s, cost in enumerate(marginal_costs):
                    obj += cost * ps[g, t, s]
    
    # Load shedding penalty
    for t in range(T):
        obj += penalty * load_shed[t]
    
    model += obj
    
    # === CONSTRAINTS ===
    
    # Thermal generator constraints
    for g in input_data['Generators']:
        if input_data['Generators'][g]['Type'] == 'Thermal':
            curve_power = input_data['Generators'][g]['Production cost curve (MW)']
            pmin, pmax = curve_power[0], curve_power[-1]
            segments_power = np.diff(curve_power)
            ramp_up = input_data['Generators'][g]['Ramp up limit (MW)']
            ramp_down = input_data['Generators'][g]['Ramp down limit (MW)']
            ramp_start = input_data['Generators'][g]['Startup limit (MW)']
            ramp_shutdown = input_data['Generators'][g]['Shutdown limit (MW)']
            init_power = input_data['Generators'][g]['Initial power (MW)']
            init_status = input_data['Generators'][g]['Initial status (h)']
            
            for t in range(T):
                commit = output_data['Is on'][g][t]
                commit_prev = 1 if init_status > 0 else 0 if t == 0 else output_data['Is on'][g][t-1]
                startup = max(0, commit - commit_prev)
                
                # p_above equals sum of segments
                model += p_above[g, t] == pl.lpSum(ps[g, t, s] for s in range(len(segments_power))), f"segment_sum_{g}_{t}"
                
                # Startup limit: p_above <= (pmax - pmin) * commit - max(0, pmax - ramp_start) * startup
                model += p_above[g, t] <= (pmax - pmin) * commit - max(0, pmax - ramp_start) * startup, f"startup_limit_{g}_{t}"
                
                # Shutdown limit (for next period)
                if t < T - 1:
                    commit_next = output_data['Is on'][g][t+1]
                    shutdown_next = max(0, commit - commit_next)
                    model += p_above[g, t] <= (pmax - pmin) * commit - max(0, pmax - ramp_shutdown) * shutdown_next, f"shutdown_limit_{g}_{t}"
                
                # Ramp constraints
                if t == 0:
                    if init_power > 0:
                        model += p_above[g, t] >= init_power - pmin - ramp_down, f"ramp_down_{g}_{t}"
                        model += p_above[g, t] <= init_power - pmin + ramp_up, f"ramp_up_{g}_{t}"
                else:
                    model += p_above[g, t] - p_above[g, t-1] >= -ramp_down, f"ramp_down_{g}_{t}"
                    model += p_above[g, t] - p_above[g, t-1] <= ramp_up, f"ramp_up_{g}_{t}"
    
    # Storage dynamics
    for s in input_data['Storage units']:
        eff_charge = input_data['Storage units'][s]['Charge efficiency']
        eff_discharge = input_data['Storage units'][s]['Discharge efficiency']
        init_level = input_data['Storage units'][s]['Initial level (MWh)']
        
        for t in range(T):
            if t == 0:
                model += level[s, t] == init_level + eff_charge * charge[s, t] - discharge[s, t] / eff_discharge, f"storage_{s}_{t}"
            else:
                model += level[s, t] == level[s, t-1] + eff_charge * charge[s, t] - discharge[s, t] / eff_discharge, f"storage_{s}_{t}"
    
    # Power balance
    for t in range(T):
        demand = input_data['Buses']['b1']['Load (MW)'][t]
        
        # Thermal generation: p = pmin * commit + p_above
        thermal_gen = 0
        for g in input_data['Generators']:
            if input_data['Generators'][g]['Type'] == 'Thermal':
                pmin = input_data['Generators'][g]['Production cost curve (MW)'][0]
                commit = output_data['Is on'][g][t]
                thermal_gen += pmin * commit + p_above[g, t]
        
        profiled_gen = pl.lpSum(profiled[g, t] for g in input_data['Generators'] if input_data['Generators'][g]['Type'] == 'Profiled')
        storage_net = pl.lpSum(discharge[s, t] - charge[s, t] for s in input_data['Storage units'])
        
        model += thermal_gen + profiled_gen + storage_net + load_shed[t] == demand, f"power_balance_{t}"
    
    # === SOLVE ===
    
    solver = pl.PULP_CBC_CMD(msg=0)
    model.solve(solver)
    
    # === EXTRACT SOLUTION ===
    
    if model.status != pl.LpStatusOptimal:
        return None
    
    solution = {
        'objective': pl.value(model.objective),
        'load_shed': [pl.value(load_shed[t]) for t in range(T)],
    }
    
    return solution


def compute_pmin_cost(input_data, output_data, T):
    """Compute minimum power production cost."""
    total_pmin_cost = 0.0
    for g in input_data['Generators']:
        if input_data['Generators'][g]['Type'] == 'Thermal':
            cost_pmin = input_data['Generators'][g]['Production cost curve ($)'][0]
            for t in range(T):
                commit = output_data['Is on'][g][t]
                total_pmin_cost += cost_pmin * commit
    return total_pmin_cost


def compute_startup_cost(input_data, output_data, T):
    """Compute startup cost."""
    total_startup_cost = 0.0
    for g in input_data['Generators']:
        if input_data['Generators'][g]['Type'] == 'Thermal':
            startup_cost = input_data['Generators'][g]['Startup costs ($)']
            startup_delays = input_data['Generators'][g]['Startup delays (h)']
            initial_status = input_data['Generators'][g]['Initial status (h)']
            hours_off = -initial_status if initial_status < 0 else 0
            for t in range(T):
                commit = output_data['Is on'][g][t]
                startup = output_data['Switch on'][g][t]
                if startup == 1:
                    cost = startup_cost[-1]
                    for k, delay in enumerate(startup_delays):
                        if hours_off < delay:
                            cost = startup_cost[k-1]
                            break
                    total_startup_cost += cost
                    hours_off = 0
                if commit == 0:
                    hours_off += 1
                else:
                    hours_off = 0
    return total_startup_cost


def compare_with_julia(solution, input_data, output_data, T):
    """Compare PuLP LP solution with Julia MILP solution."""
    # Compute costs to match Julia's objective
    pmin_cost = compute_pmin_cost(input_data, output_data, T)
    startup_cost = compute_startup_cost(input_data, output_data, T)
    
    # PuLP total = variable cost (from objective) + pmin cost + startup cost
    pulp_total = solution['objective'] + pmin_cost + startup_cost
    
    # Julia total = thermal cost + load curtail cost + startup cost
    julia_thermal = sum(sum(costs) for costs in output_data['Thermal production cost ($)'].values())
    penalty = input_data['Parameters']['Power balance penalty ($/MW)']
    julia_curtail = sum(sum(curtails) for curtails in output_data['Load curtail (MW)'].values())
    julia_total = julia_thermal + julia_curtail * penalty + startup_cost
    
    # Show comparison
    print(f"\nComparison:")
    print(f"  PuLP (LP):     ${pulp_total:,.2f}")
    print(f"  Julia (MILP):  ${julia_total:,.2f}")
    print(f"  Difference:    {100*(pulp_total - julia_total)/julia_total:+.4f}%")


if __name__ == "__main__":
    # Find all instance directories
    base_path = Path(".")
    instances = sorted([d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("instance_")])
    
    print(f"Found {len(instances)} instances\n")
    print("=" * 80)
    
    results = []
    
    for instance_path in instances:
        instance_name = instance_path.name
        print(f"\nProcessing: {instance_name}")
        
        try:
            # Load data
            input_path = instance_path / "InputData.json.gz"
            with gzip.open(input_path, 'rt') as f:
                input_data = json.load(f)
            output_path = instance_path / "OutputData.json.gz"
            with gzip.open(output_path, 'rt') as f:
                output_data = json.load(f)
            T = input_data['Parameters']['Time horizon (h)']
            
            # Solve
            solution = solve_economic_dispatch(instance_name)
            
            if solution:
                # Compute costs
                pmin_cost = compute_pmin_cost(input_data, output_data, T)
                startup_cost = compute_startup_cost(input_data, output_data, T)
                pulp_total = solution['objective'] + pmin_cost + startup_cost
                
                julia_thermal = sum(sum(costs) for costs in output_data['Thermal production cost ($)'].values())
                penalty = input_data['Parameters']['Power balance penalty ($/MW)']
                julia_curtail = sum(sum(curtails) for curtails in output_data['Load curtail (MW)'].values())
                julia_total = julia_thermal + julia_curtail * penalty + startup_cost
                
                diff_percent = 100 * (pulp_total - julia_total) / julia_total
                
                results.append({
                    'instance': instance_name,
                    'pulp_cost': pulp_total,
                    'julia_cost': julia_total,
                    'difference_pct': diff_percent
                })
                
                print(f"  PuLP (LP):     ${pulp_total:,.2f}")
                print(f"  Julia (MILP):  ${julia_total:,.2f}")
                print(f"  Difference:    {diff_percent:+.4f}%")
            else:
                print(f"  ❌ Failed to solve")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    # Summary
    print("\n" + "=" * 80)
    print("\nSUMMARY OF ALL INSTANCES")
    print("=" * 80)
    print(f"{'Instance':<30} {'PuLP Cost':>15} {'Julia Cost':>15} {'Diff %':>10}")
    print("-" * 80)
    
    for r in results:
        print(f"{r['instance']:<30} ${r['pulp_cost']:>14,.2f} ${r['julia_cost']:>14,.2f} {r['difference_pct']:>9.4f}%")
    
    if results:
        avg_diff = sum(r['difference_pct'] for r in results) / len(results)
        print("-" * 80)
        print(f"{'Average difference:':<30} {'':<15} {'':<15} {avg_diff:>9.4f}%")