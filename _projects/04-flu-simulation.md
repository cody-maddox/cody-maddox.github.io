---
title: "Classroom Flu Simulation"
summary: "Stochastic epidemiological model simulating pandemic flu transmission in a 61-student classroom using EpiModel in R, with and without immunity scenarios across 1,000 trials."
tags: [Simulation, Epidemiology, R, Statistics]
order: 5
---

## Overview

This project models the spread of pandemic flu within a classroom of 61 students using stochastic simulation. Two scenarios are compared: a baseline where all students are susceptible, and a vaccination scenario where each student has a 50% chance of being perfectly immunized. Each scenario is simulated 1,000 times to capture the range of possible outcomes.

## Scenarios

### No Immunity Baseline
All 61 students are susceptible. The simulation tracks how quickly the infection spreads and how many students are ultimately infected.

### 50% Immunity
Each student independently has a 50% probability of being perfectly immunized before the simulation begins. This models a realistic partial-vaccination scenario and demonstrates herd immunity effects.

## Model Parameters

- **Population:** 61 students in a closed classroom
- **Recovery:** Probabilistic with a daily recovery rate of 1/3
- **Replications:** 1,000 trials per scenario
- **Framework:** SIR (Susceptible-Infected-Recovered) model via the EpiModel package

## Results

### No Immunity — Expected Infections Over Time
![Expected infections without immunity]({{ '/assets/images/flu_expected_number_of_kids_infected.png' | relative_url }})

### No Immunity — Epidemic Duration Distribution
![Epidemic duration histogram]({{ '/assets/images/flu_duration_histogram.png' | relative_url }})

### With Immunity — Expected Infections Over Time
![Expected infections with immunity]({{ '/assets/images/expected_infections_immunity.png' | relative_url }})

### With Immunity — Epidemic Duration Distribution
![Epidemic duration with immunity]({{ '/assets/images/epidemic_immunity_histo.png' | relative_url }})

## Key Findings

- Without immunity, the infection spreads rapidly through the entire classroom
- With 50% immunity, epidemics are shorter and infect significantly fewer students
- The stochastic nature produces a wide range of outcomes — some trials see early burnout while others see full-classroom infection
- The probabilistic recovery model (daily rate of 1/3) better reflects real-world flu recovery than a fixed duration

## Tools & Libraries

R, EpiModel, ggplot2
