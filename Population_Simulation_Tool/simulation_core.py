# === simulation_core.py ===
import random
import math
import numpy as np
import pandas as pd
from tqdm import tqdm

class Event:
    def __init__(self, name, start_year, end_year, effects):
        self.name = name
        self.start_year = start_year
        self.end_year = end_year
        self.effects = effects

    def is_active(self, year):
        return self.start_year <= year <= self.end_year

class GlobalFactors:
    def __init__(self):
        self.unemployment = 0.06
        self.econ_policy_index = 1.0

    def update(self, year):
        if year < 1980:
            self.unemployment = 0.04
            self.econ_policy_index = 1.1
        elif year < 2000:
            self.unemployment = 0.07
            self.econ_policy_index = 1.0
        else:
            self.unemployment = 0.09
            self.econ_policy_index = 0.9

class Person:
    def __init__(self, sex, urban=True):
        self.sex = sex
        self.age = 0
        self.partner = None
        self.children = []
        self.copulated = False
        self.education = random.uniform(0, 1)
        self.income = random.uniform(0.2, 1.0)
        self.urban = urban

    def fertility(self):
        if self.sex == 'f':
            if self.age < 20: return 0.8
            elif 20 <= self.age <= 30: return 1.0
            elif 31 <= self.age <= 35: return 0.8
            elif 36 <= self.age <= 40: return 0.5
            elif 41 <= self.age <= 45: return 0.2
            else: return 0.01
        else:
            if self.age < 20: return 0.9
            elif 20 <= self.age <= 35: return 1.0
            elif 36 <= self.age <= 50: return 0.8
            elif 51 <= self.age <= 65: return 0.3
            else: return 0.1

    def mortality(self, healthcare_quality=0.8):
        a, b, c, d, e = 0.005, 1, 0.0001, 0.0001, 0.077
        base = a * math.exp(-b * self.age) + c + d * math.exp(e * self.age)
        m1 = 1 - (healthcare_quality * 0.5)
        m2 = 1 - ((self.income + self.education) / 4)
        return base * m1 * m2

def conception_prob(person, partner, child_support=0.0, education_impact=0.5):
    fertility = person.fertility() * partner.fertility()
    n_children = len(person.children)
    base_prob = min(fertility * 0.25 * 12, 1) / (n_children + 1)
    education_modifier = 1 - (person.education * education_impact)
    if not person.urban:
        education_modifier = 1 - (person.education * education_impact * 0.5)
    income_modifier = 1 + (person.income * 0.1) + (child_support * (n_children + 1))
    return base_prob * education_modifier * income_modifier

def run_simulation(init_pop_count=1000, n_years=100, urban_ratio=0.6,
                   base_child_support=0.0, base_education_impact=0.5,
                   base_healthcare_quality=0.8, events=None):

    if events is None:
        events = []

    global_factors = GlobalFactors()
    population = [Person('m', random.random() < urban_ratio) for _ in range(init_pop_count // 2)]
    population += [Person('f', random.random() < urban_ratio) for _ in range(init_pop_count // 2)]

    init_ages = [max(int(random.normalvariate(30, 20)), 0) for _ in population]
    for p, a in zip(population, init_ages): p.age = a

    pop_sizes, child_bearing_ages = [], []
    urban_population, rural_population, avg_education, dependency_ratios = [], [], [], []
    age_distributions = []

    for year in tqdm(range(n_years)):
        global_factors.update(year)
        active_params = {
            "child_support": base_child_support,
            "education_impact": base_education_impact[year] if isinstance(base_education_impact, list) else base_education_impact,
            "healthcare_quality": base_healthcare_quality
        }
        for event in events:
            if event.is_active(year):
                active_params.update(event.effects)

        immigration_inflow = active_params.get("immigration_inflow", 0)
        for _ in range(immigration_inflow // 2):
            for sex in ['m', 'f']:
                person = Person(sex, random.random() < urban_ratio)
                person.age = max(int(random.normalvariate(25, 10)), 18)
                person.education = random.uniform(0.2, 0.8)
                person.income = random.uniform(0.2, 0.6)
                population.append(person)

        pop_sizes.append(len(population))
        urban_population.append(sum(1 for p in population if p.urban))
        rural_population.append(len(population) - urban_population[-1])
        avg_education.append(sum(p.education for p in population) / len(population))
        current_ages = [p.age for p in population]
        age_distributions.append(current_ages)
        working = sum(1 for p in population if 15 <= p.age <= 64)
        young = sum(1 for p in population if p.age < 15)
        old = sum(1 for p in population if p.age > 64)
        dependency_ratios.append((young + old) / working if working else 0)

        babies = []
        random.shuffle(population)
        for i in range(len(population) - 1):
            a, b = population[i], population[i+1]
            if a.age >= 18 and not a.partner and b.age >= 18 and not b.partner and a.sex != b.sex:
                a.partner, b.partner = b, a
            elif a.partner and not a.copulated:
                a.copulated, a.partner.copulated = True, True
                female = a if a.sex == 'f' else a.partner
                male = a.partner if a.sex == 'f' else a
                prob = conception_prob(female, male, active_params["child_support"], active_params["education_impact"])
                if random.random() < prob:
                    baby = Person('m' if random.random() < 0.5 else 'f', female.urban)
                    baby.education = max(0, min(1, (female.education + male.education) / 2 + random.uniform(-0.1, 0.1)))
                    baby.income = max(0.2, min(1, (female.income + male.income) / 2 + random.uniform(-0.2, 0.2)))
                    child_bearing_ages += [female.age, male.age]
                    female.children.append(baby)
                    male.children.append(baby)
                    babies.append(baby)

        deaths = []
        for p in population:
            if random.random() < p.mortality(active_params["healthcare_quality"]):
                if p.partner: p.partner.partner = None
                deaths.append(p)
            else:
                p.age += 1
                p.copulated = False
                if 18 <= p.age <= 65:
                    p.income = min(1.0, p.income + random.uniform(-0.05, 0.1) * global_factors.econ_policy_index)

        population = [p for p in population if p not in deaths] + babies
        for p in population:
            if 5 <= p.age <= 25:
                p.education = min(1.0, max(0.0, p.education + 0.02 * active_params["education_impact"]))


    final_ages = [p.age for p in population]

    return {
        "population": population,
        "pop_sizes": pop_sizes,
        "init_ages": init_ages,
        "final_ages": final_ages,
        "child_bearing_ages": child_bearing_ages,
        "urban_population": urban_population,
        "rural_population": rural_population,
        "avg_education": avg_education,
        "age_distributions": age_distributions,
        "dependency_ratios": dependency_ratios
    }
