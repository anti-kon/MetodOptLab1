import copy

import eel

import logic.potentials_method as pm

eel.init('web')

@eel.expose
def get_brute_force_result(cost_matrix, proposal_vector, demand_vector):
    save_cost_matrix = copy.deepcopy(cost_matrix)
    save_proposal_vector = copy.deepcopy(proposal_vector)
    save_demand_vector = copy.deepcopy(demand_vector)
    canonical_matrix, basis, function_vector, x_limits, matrix_equality, matrix_less = pm.get_canonical(
        cost_matrix, proposal_vector, demand_vector)

    save_matrix = copy.deepcopy(canonical_matrix)
    save_basis = copy.deepcopy(basis)
    brute_force = pm.brute_force(canonical_matrix, function_vector, basis, x_limits, matrix_equality, [], matrix_less)
    answer_vector = brute_force[0]
    answer = 0

    for i in range(0, len(answer_vector)):
        answer += function_vector[i] * answer_vector[i]

    local_matrix, local_function, local_basis = pm.getDouble(canonical_matrix, function_vector, basis)

    print(brute_force)
    return save_cost_matrix, save_proposal_vector, save_demand_vector, answer_vector, answer, save_matrix, save_basis, function_vector, local_matrix, local_function, local_basis, brute_force[1]


@eel.expose
def get_potentials_method_result(cost_matrix, proposal_vector, demand_vector):
    save_cost_matrix = copy.deepcopy(cost_matrix)
    save_proposal_vector = copy.deepcopy(proposal_vector)
    save_demand_vector = copy.deepcopy(demand_vector)

    advantage = pm.sum_balance_check(proposal_vector, demand_vector)
    if advantage < 0:
        proposal_vector.append(abs(advantage))
        cost_matrix.append([0] * len(cost_matrix[0]))
    elif advantage > 0:
        demand_vector.append(abs(advantage))
        for i in range(0, len(cost_matrix)):
            cost_matrix[i].append(0)

    is_optimal = False
    path_matrix, matrix_mask = pm.northwest_method(cost_matrix, proposal_vector, demand_vector)
    start_path_matrix = copy.deepcopy(path_matrix)
    start_matrix_mask = copy.deepcopy(matrix_mask)

    solve_steps = []
    step_start_matrix = copy.deepcopy(path_matrix)
    step_matrix_mask = copy.deepcopy(matrix_mask)
    while not is_optimal:
        print(path_matrix, matrix_mask)
        is_optimal, path_matrix, matrix_mask, proposal_potentials, demand_potentials, path_deltas, cycle = (
            pm.potentials_method(path_matrix, cost_matrix, matrix_mask))
        solve_steps.append([copy.deepcopy(step_start_matrix), copy.deepcopy(step_matrix_mask),
                            copy.deepcopy(proposal_potentials), copy.deepcopy(demand_potentials),
                            copy.deepcopy(path_deltas), copy.deepcopy(cycle)])
        step_start_matrix = copy.deepcopy(path_matrix)
        step_matrix_mask = copy.deepcopy(matrix_mask)

    return (save_cost_matrix, save_proposal_vector, save_demand_vector, advantage == 0, cost_matrix, proposal_vector,
            demand_vector, start_path_matrix, start_matrix_mask, solve_steps, path_matrix)


eel.start('index.html')

