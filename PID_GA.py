import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import os
import pandas as pd

# parameter sistem motor
J = 0.01  #moment of inertia
b = 0.1   #damping ratio
K = 0.01  #motor constant
setpoint = 1000

# parameter GA
population_size = 6
num_generations = 50
mutation_rate = 0.75
crossover_rate = 0.25
population = [np.random.rand(3) for _ in range(population_size)]

# setting waktu
t = np.linspace(0, 10, 1000)
dt = t[1] - t[0]


def motor_speed(x, t, u):
    theta, omega = x
    dxdt = [omega, (u - b*omega - K*theta)/J]
    return dxdt


def pid_controller(y, setpoint, integral, last_error, dt, Kp, Ki, Kd):
    error = setpoint - y
    integral += error * dt
    derivative = (error - last_error) / dt
    output = Kp * error + Ki * integral + Kd * derivative
    return output, integral, error #nilai akumulasi error dan error juga dikembalikan


def selection(population, fitnesses, t_size=3):
    selected_indices = np.random.choice(range(len(population)), t_size, replace=False)
    selected_fitnesses = [fitnesses[idx][1] for idx in selected_indices] 
    winner_index = selected_indices[np.argmin(selected_fitnesses)] #semakin kecil fitness, semakin bagus individu, argmin
    return population[winner_index]

def crossover(individual1, individual2):
    if np.random.rand() < crossover_rate:
        point = np.random.randint(1, len(individual1) - 1)
        return (np.concatenate([individual1[:point], individual2[point:]]),
                np.concatenate([individual2[:point], individual1[point:]]))
    else:
        return individual1, individual2

def mutate(individual):
    if np.random.rand() < mutation_rate:
        mutation_point = np.random.randint(0, len(individual))
        individual[mutation_point] += np.random.normal(-1, 1)
        individual[mutation_point] = max(0, individual[mutation_point]) #memastikan selalu +
    return individual

def evaluate(gen, indiv_id, individual, record):
    Kp, Ki, Kd = individual
    y0 = [0, 0]
    y = [0]
    integral_err = 0
    last_err = 0
    for i in range(1, len(t)):
        u, integral, err = pid_controller(y[-1], setpoint, integral_err, last_err, dt, Kp, Ki, Kd)
        integral_err = integral
        last_err = err
        next_point = odeint(motor_speed, y0, [t[i-1], t[i]], args=(u,))[-1]
        y0 = next_point
        y.append(next_point[0])


    rmse = np.sqrt(np.mean((setpoint - np.array(y))**2))
    overshoot = (max(y) - setpoint) / setpoint * 100
    rise_time_indices = np.where(np.array(y) > setpoint * 0.9)[0]
    rise_time = t[rise_time_indices[0]] if rise_time_indices.size > 0 else 10000 #np.nan
    settling_time_indices = np.where(np.abs(np.array(y) - setpoint) < setpoint * 0.02)[0]
    settling_time = t[settling_time_indices[0]] if settling_time_indices.size > 0 else 10000 #np.nan


    fitness = (0.7*rmse) + (0.1*overshoot) + (0.1*rise_time) + (0.1*settling_time)

    # Record 
    if record:
        foldername="motor_speed/"
        os.makedirs(foldername, exist_ok=True)
        filename=format(gen, '03d')+'_'+format(indiv_id, '03d')
        plt.figure()
        plt.plot(np.array(y)) 
        plt.ylim(0, setpoint+(0.35*setpoint)) 
        plt.grid(True)  
        plt.title('Motor Speed') 
        plt.xlabel('Time') 
        plt.ylabel('RPM')  
        plt.savefig(foldername+filename+'.png')
        plt.close() 

    return filename, fitness, rmse, overshoot, rise_time, settling_time#, np.array(y)


records = []
for generation in range(num_generations):
    print("========================================")
    print("Gen: ", generation)

    fitnesses = []
    idx = 0
    for individual in population:
        filename, fitness, rmse, overshoot, rise_time, settling_time = evaluate(generation, idx, individual, record=True) #, speed
        fitnesses.append((filename, fitness, rmse, overshoot, rise_time, settling_time, individual))
        idx=idx+1

        print("---------------------------------------")
        print("(Kp, Ki, Kd): ", individual)
        print("fitness: ", fitness)
        print("rmse: ", rmse)
        print("overshoot: ", overshoot)
        print("rise_time: ", rise_time)
        print("settling_time: ", settling_time)
        
    for filename, fitness, rmse, overshoot, rise_time, settling_time, individual in fitnesses:
        records.append([generation] + individual.tolist() + [filename, fitness, rmse, overshoot, rise_time, settling_time])
    
    # selection, crossover, mutation
    new_population = []
    while len(new_population) < population_size:
        parent1 = selection(population, fitnesses)
        new_population.append(parent1)
        parent2 = selection(population, fitnesses)
        new_population.append(parent2)
        child1, child2 = crossover(parent1, parent2)
        new_population.append(child1)
        new_population.append(child2)
        new_population.append(mutate(child1))
        new_population.append(mutate(child2))
    
    population = new_population
    print("========================================")


df = pd.DataFrame(records, columns=['Generation', 'Kp', 'Ki', 'Kd', 'filename', 'Fitness', 'RMSE', 'Overshoot', 'RiseTime', 'SettlingTime'])
df.to_excel('pid_ga_optimization_records.xlsx', index=False)

