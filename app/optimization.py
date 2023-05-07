from copy import deepcopy

import numpy as np
import streamlit as st
from create_backtest import financial_evaluation, qs_metrics


def check_constraints(params, constraints):
    if len(params) != len(constraints):
        raise ValueError("Length of parameters and constraints should be same")
    return not any(
        params[i] > constraints[i][1] or params[i] < constraints[i][0]
        for i in range(len(params))
    )


def fitness(ohlcv, predictions, metric_optimized, individual):
    portfolio, benchmark, charts_dict_params = financial_evaluation(
        ohlcv=ohlcv,
        predictions=predictions,
        **st.session_state["backtest_configuration"],
        take_profit=individual[0],
        stop_loss=individual[1],
        leverage=individual[2],
        show_initial_configuration=False,
        show_tables=False,
        show_charts=False,
        show_time=False,
    )
    strategy_returns = portfolio["Value"].pct_change().dropna()
    benchmark_returns = benchmark["Close"].pct_change().dropna()
    metrics_dict, metrics_df = qs_metrics(
        strategy_returns, benchmark_returns, risk_free_rate=1
    )
    return metrics_df["Strategy"][metric_optimized]


def roulette_wheel_selection(p):
    c = np.cumsum(p)
    r = sum(p) * np.random.rand()
    ind = np.argwhere(r <= c)
    return ind[0][0]


def gene_crossover(parent1, parent2):
    child1 = deepcopy(parent1)
    child2 = deepcopy(parent2)
    for i in range(len(parent1)):
        prob = np.random.uniform(0, 1, 1)
        if prob < 0.5:
            child1[i] = parent2[i]
        else:
            child2[i] = parent1[i]
    return child1, child2


def mutate(individual, constraints, mutation_probability):
    MUTATION_RANGE = [
        (-15, 15),
        (-15, 15),
        (-5, 5),
    ]
    MUTATION_PROB = mutation_probability
    mutated = deepcopy(individual)
    for i in range(len(individual)):
        prob = np.random.uniform(0, 1, 1)
        if prob < MUTATION_PROB:
            mutated[i] = individual[i] + np.random.randint(
                MUTATION_RANGE[i][0], MUTATION_RANGE[i][1]
            )
            iteration = 0
            while not check_constraints(mutated, constraints):
                mutated[i] = individual[i] + np.random.randint(
                    MUTATION_RANGE[i][0], MUTATION_RANGE[i][1]
                )
                iteration += 1
                if iteration > 1000:
                    return individual
    return mutated


def optimize(
    col,
    ohlcv,
    predictions,
    metric_optimized,
    take_profit_values,
    stop_loss_values,
    leverage_values,
    population_size,
    mutation_probability,
    iteration,
):
    del st.session_state["backtest_configuration"]["take_profit"]
    del st.session_state["backtest_configuration"]["stop_loss"]
    del st.session_state["backtest_configuration"]["leverage"]
    POPULATION_SIZE = population_size
    ITERATIONS = iteration
    PARAMS = [
        np.mean(take_profit_values),
        np.mean(stop_loss_values),
        np.mean(leverage_values),
    ]
    CONSTRAINTS = [
        (take_profit_values[0], take_profit_values[1], "int"),
        (stop_loss_values[0], stop_loss_values[1], "int"),
        (leverage_values[0], leverage_values[1], "int"),
    ]
    population = np.array([list(range(len(PARAMS)))])

    with col:
        with st.spinner("Creating the initial population..."):
            for _ in range(POPULATION_SIZE):
                params = np.array([])
                for i in range(len(PARAMS)):
                    random_param = np.random.uniform(
                        CONSTRAINTS[i][0], CONSTRAINTS[i][1]
                    )
                    if CONSTRAINTS[i][2] == "int":
                        random_param = int(random_param)
                    params = np.append(params, random_param)
                population = np.vstack([population, params])

            population = population[1:]
            scores = np.array([])

            for individual in population:
                # kendi backtestinin sonuclari
                (
                    portfolio,
                    benchmark,
                    charts_dict_params,
                ) = financial_evaluation(
                    ohlcv=ohlcv,
                    predictions=predictions,
                    **st.session_state["backtest_configuration"],
                    take_profit=individual[0],
                    stop_loss=individual[1],
                    leverage=individual[2],
                    show_initial_configuration=False,
                    show_tables=False,
                    show_charts=False,
                    show_time=False,
                )

                strategy_returns = portfolio["Value"].pct_change().dropna()
                benchmark_returns = benchmark["Close"].pct_change().dropna()

                metrics_dict, metrics_df = qs_metrics(
                    strategy_returns, benchmark_returns, risk_free_rate=1
                )
                scores = np.append(
                    scores, metrics_df["Strategy"][metric_optimized]
                )
    t = st.empty()
    beta = 1
    for i in range(ITERATIONS):
        best_return = max(scores)
        output = f"Iteration: {i}, Best value of {metric_optimized} so far: {best_return}"
        t.markdown(f"## {output} ...")

        scores = np.array(scores)
        avg_cost = np.mean(scores)
        if avg_cost != 0:
            a = scores / avg_cost
        probs = np.exp(beta * a)

        parent1 = population[roulette_wheel_selection(probs)]
        parent2 = population[roulette_wheel_selection(probs)]
        child1, child2 = gene_crossover(parent1, parent2)

        child1 = mutate(child1, CONSTRAINTS, mutation_probability)
        child2 = mutate(child2, CONSTRAINTS, mutation_probability)

        fitness1 = fitness(ohlcv, predictions, metric_optimized, child1)
        fitness2 = fitness(ohlcv, predictions, metric_optimized, child2)
        if fitness1 > best_return:
            index = np.random.randint(0, len(population))
            population = np.delete(population, index, 0)
            scores = np.delete(scores, index)
            population = np.vstack([population, child1])
            scores = np.append(scores, fitness1)
        if fitness2 > best_return:
            index = np.random.randint(0, len(population))
            population = np.delete(population, index, 0)
            scores = np.delete(scores, index)
            population = np.vstack([population, child2])
            scores = np.append(scores, fitness2)
    best_params = population[np.where(scores == max(scores))[0]][0]
    result = f"Best parameters: {best_params}, \
        Best value of {metric_optimized}: {max(scores)} "
    t.markdown(f"## {result}")

    st.session_state["backtest_configuration"][
        "optimized_take_profit"
    ] = best_params[0]
    st.session_state["backtest_configuration"][
        "optimized_stop_loss"
    ] = best_params[1]
    st.session_state["backtest_configuration"][
        "optimized_leverage"
    ] = best_params[2]
    st.success("Best parameters added to the strategy.")
