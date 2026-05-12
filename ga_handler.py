import os
import json
import random
import time
import logging
from decimal import Decimal

import boto3
import numpy as np
from deap import base, creator, tools
from aws_xray_sdk.core import xray_recorder
from aws_xray_sdk.core import patch_all

patch_all()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

REGION = os.environ.get("AWS_REGION", "eu-west-1")
ENDPOINT_NAME = os.environ["SAGEMAKER_ENDPOINT_NAME"]
TABLE_NAME = os.environ["DYNAMODB_TABLE_NAME"]

sm_runtime = boto3.client("sagemaker-runtime", region_name=REGION)
dynamodb = boto3.resource("dynamodb", region_name=REGION)
table = dynamodb.Table(TABLE_NAME)
cloudwatch = boto3.client("cloudwatch", region_name=REGION)

# GA Configuration
GA_CONFIG = {
    "POPULATION_SIZE": 1000,
    "GENERATIONS": 150,
    "ELITISM_COUNT": 10,
    "CROSSOVER_PROB": 0.7,
    "MUTATION_PROB": 0.3,
    "MUTATION_STRENGTH": 0.2,
    "TOURNAMENT_SIZE": 3,
    "MIN_INGREDIENT_PCT": 0.01,
    "MAX_INGREDIENT_PCT": 0.35,
    "MIN_MANDATORY_PCT": 0.15,
    "TOTAL_PCT_TARGET": 1.0,
    "TOTAL_PCT_TOLERANCE": 0.1,
    "FLAVOR_WEIGHT": 0.8,
    "SIMPLICITY_WEIGHT": 0.2,
    "PENALTY_STRENGTH": 0.7,
}

FLAVOR_CATEGORIES = ["bittersweet", "citrus", "creamy", "floral", "fruity", "herbal", "savoury", "spicy", "sweet"]

OPTIMAL_THRESHOLDS = {
    "bittersweet": 0.50,
    "citrus": 0.50,
    "creamy": 0.60,
    "floral": 0.90,
    "fruity": 0.60,
    "herbal": 0.55,
    "savoury": 0.80,
    "spicy": 0.75,
    "sweet": 0.80
}

INGREDIENT_COLUMNS = None

def getIngridientColumns():
    global INGREDIENT_COLUMNS
    if INGREDIENT_COLUMNS is None:
        s3 = boto3.client("s3", region_name=REGION)
        response = s3.get_object(
            Bucket="cocktail-ai",
            Key="ingridients/feature_columns.json"
        )
        INGREDIENT_COLUMNS = json.loads(response["Body"].read().decode("utf-8"))
        logger.info(f"Loaded {len(INGREDIENT_COLUMNS)} ingredient columns from S3")
    return INGREDIENT_COLUMNS

# It throwed an error when I tried to store python float, so converting it to Decimal.
def toDecimal(value):
    return Decimal(str(round(float(value), 6)))

# We store the recipe json in dynamoDB, hence we need to convert all floats to Decimal in that object.
def sanitiseDynamoData(obj):
    if isinstance(obj, float) or isinstance(obj, np.floating):
        return toDecimal(obj)
    if isinstance(obj, dict):
        return {k: sanitiseDynamoData(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitiseDynamoData(i) for i in obj]
    return obj

# As mentioned in the report, this function used to update the job status in DynamoDB.
def updateJobStatus(job_id: str, status: str, extra: dict = None):
    update_data = {"status": status, "updatedAt": int(time.time())}
    if extra:
        update_data.update(sanitiseDynamoData(extra))

    update_expr = "SET " + ", ".join(f"#k{i} = :v{i}" for i in range(len(update_data)))
    expr_names = {f"#k{i}": k for i, k in enumerate(update_data.keys())}
    expr_values = {f":v{i}": v for i, v in enumerate(update_data.values())}

    table.update_item(
        Key={"jobId": job_id},
        UpdateExpression=update_expr,
        ExpressionAttributeNames=expr_names,
        ExpressionAttributeValues=expr_values,
    )
    logger.info(f"DynamoDB jobId={job_id}  status={status}")

def callSagemakerEndpoint(population: list) -> dict:
    payload = json.dumps({"population": population})

    # Added Xray here too, so that we can monitor every API call.
    with xray_recorder.in_subsegment("SageMaker.InvokeEndpoint") as subsegment:
        subsegment.put_metadata("population_size", len(population))
        response = sm_runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Accept="application/json",
            Body=payload,
        )
        body = response["Body"].read().decode("utf-8")
        subsegment.put_metadata("response_size_bytes", len(body))

    return json.loads(body)

def normalize(individual):
    total = sum(individual)
    if total > 0:
        factor = GA_CONFIG["TOTAL_PCT_TARGET"] / total
        for i in range(len(individual)):
            individual[i] *= factor
    return individual

def makeIndividual(ingredient_columns, mandatory_indices=None, mandatory_percentages=None):
    ind = [0.0] * len(ingredient_columns)
    if mandatory_indices:
        for i, idx in enumerate(mandatory_indices):
            pct = (mandatory_percentages[i] if mandatory_percentages and i < len(mandatory_percentages)
                   else random.uniform(GA_CONFIG["MIN_MANDATORY_PCT"], GA_CONFIG["MAX_INGREDIENT_PCT"]))
            ind[idx] = pct

    available = [i for i in range(len(ind)) if i not in (mandatory_indices or [])]
    n_optional = random.randint(0, 10)
    for idx in random.sample(available, min(n_optional, len(available))):
        ind[idx] = random.uniform(GA_CONFIG["MIN_INGREDIENT_PCT"], GA_CONFIG["MAX_INGREDIENT_PCT"])

    normalize(ind)
    return creator.Individual(ind)

def cxBlend(ind1, ind2):
    for i in range(len(ind1)):
        if random.random() < 0.5:
            alpha = random.uniform(-0.5, 1.5)
            ind1[i], ind2[i] = (
                alpha * ind1[i] + (1 - alpha) * ind2[i],
                alpha * ind2[i] + (1 - alpha) * ind1[i]
            )
    for ind in (ind1, ind2):
        for i in range(len(ind)):
            ind[i] = max(0.0, min(GA_CONFIG["MAX_INGREDIENT_PCT"], ind[i]))
        normalize(ind)
    return ind1, ind2

def mutGaussian(individual):
    for i in range(len(individual)):
        if random.random() < 0.1:
            if individual[i] > 0:
                individual[i] = max(0.0, min(GA_CONFIG["MAX_INGREDIENT_PCT"], individual[i] + random.gauss(0, GA_CONFIG["MUTATION_STRENGTH"])))
            elif random.random() < 0.05:
                individual[i] = random.uniform(GA_CONFIG["MIN_INGREDIENT_PCT"], GA_CONFIG["MAX_INGREDIENT_PCT"])

    if random.random() < 0.1:
        active = [i for i, v in enumerate(individual) if v > GA_CONFIG["MIN_INGREDIENT_PCT"]]
        if active:
            individual[random.choice(active)] = 0.0

    normalize(individual)
    return (individual,)

def evaluatePopulation(population, target_flavors, mandatory_indices):
    all_probs = callSagemakerEndpoint([list(ind) for ind in population])

    fitnesses = []
    for i, individual in enumerate(population):
        flavor_fitness = 0.0
        total_weight = 0.0
        for flavor, weight in target_flavors.items():
            if flavor not in FLAVOR_CATEGORIES:
                continue
            proba = all_probs[flavor][i]
            threshold = OPTIMAL_THRESHOLDS.get(flavor, 0.5)
            if proba < threshold:
                proba *= GA_CONFIG["PENALTY_STRENGTH"]
            flavor_fitness += proba * weight
            total_weight += weight

        if total_weight:
            flavor_fitness /= total_weight

        active_count = sum(1 for x in individual if x > GA_CONFIG["MIN_INGREDIENT_PCT"])
        simplicity_score = max(0, 8 - active_count) / 8.0

        mandatory_penalty = 0.0
        if mandatory_indices:
            for idx in mandatory_indices:
                if individual[idx] < GA_CONFIG["MIN_MANDATORY_PCT"]:
                    mandatory_penalty += 0.2

        total_pct = sum(individual)
        total_penalty = min(
            abs(total_pct - GA_CONFIG["TOTAL_PCT_TARGET"]) / GA_CONFIG["TOTAL_PCT_TOLERANCE"],
            1.0
        )

        fitness = (
            GA_CONFIG["FLAVOR_WEIGHT"] * flavor_fitness
            + GA_CONFIG["SIMPLICITY_WEIGHT"] * simplicity_score
            - mandatory_penalty
            - total_penalty
        )
        fitnesses.append((max(0.0, fitness),))

    return fitnesses

def run_ga(target_flavors: dict, mandatory_ingredients: list = None, mandatory_percentages: list = None) -> tuple:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    ingredient_columns = getIngridientColumns()
    ingredient_to_idx = {ing: i for i, ing in enumerate(ingredient_columns)}

    mandatory_indices = []
    if mandatory_ingredients:
        for name in mandatory_ingredients:
            if name in ingredient_to_idx:
                mandatory_indices.append(ingredient_to_idx[name])
            else:
                logger.warning(f"Mandatory ingredient '{name}' not found — skipping")

    toolbox = base.Toolbox()
    toolbox.register("individual", makeIndividual, ingredient_columns, mandatory_indices, mandatory_percentages)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", cxBlend)
    toolbox.register("mutate", mutGaussian)
    toolbox.register("select", tools.selTournament, tournsize=GA_CONFIG["TOURNAMENT_SIZE"])

    pop = toolbox.population(n=GA_CONFIG["POPULATION_SIZE"])
    fitnesses = evaluatePopulation(pop, target_flavors, mandatory_indices)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    hof = tools.HallOfFame(GA_CONFIG["ELITISM_COUNT"])
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("max", np.max)
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals", "avg", "max"]

    for gen in range(GA_CONFIG["GENERATIONS"]):
        offspring = list(map(toolbox.clone, toolbox.select(pop, len(pop) - GA_CONFIG["ELITISM_COUNT"])))

        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < GA_CONFIG["CROSSOVER_PROB"]:
                toolbox.mate(c1, c2)
                del c1.fitness.values
                del c2.fitness.values

        for mutant in offspring:
            if random.random() < GA_CONFIG["MUTATION_PROB"]:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid = [ind for ind in offspring if not ind.fitness.valid]
        if invalid:
            new_fits = evaluatePopulation(invalid, target_flavors, mandatory_indices)
            for ind, fit in zip(invalid, new_fits):
                ind.fitness.values = fit

        pop = tools.selBest(pop, GA_CONFIG["ELITISM_COUNT"]) + offspring
        hof.update(pop)

        record = stats.compile(pop)
        logbook.record(gen=gen, nevals=len(invalid), **record)
        logger.info(f"Gen {gen:3d} | avg={record['avg']:.3f} | best={record['max']:.3f}")

        cloudwatch.put_metric_data(
            Namespace="CocktailAI/GA",
            MetricData=[
                {
                    "MetricName": "BestFitness",
                    "Value": float(record["max"]),
                    "Unit": "None",
                    "Dimensions": [{"Name": "EndpointName", "Value": ENDPOINT_NAME}],
                },
                {
                    "MetricName": "AvgFitness",
                    "Value": float(record["avg"]),
                    "Unit": "None",
                    "Dimensions": [{"Name": "EndpointName", "Value": ENDPOINT_NAME}],
                },
            ],
        )

    best = tools.selBest(pop, 1)[0]
    logger.info(f"GA finished — best fitness: {best.fitness.values[0]:.4f}")
    return best, logbook

def formatRecipe(individual) -> list:
    ingredient_columns = getIngridientColumns()
    recipe = [
        {
            "ingredient": ingredient_columns[i],
            "percentage": round(float(pct) * 100, 2)   # float() strips numpy type
        }
        for i, pct in enumerate(individual)
        if pct > GA_CONFIG["MIN_INGREDIENT_PCT"]
    ]
    recipe.sort(key=lambda x: x["percentage"], reverse=True)
    return recipe

# Lambda Handler
def handler(event, context):
    logger.info(f"Received {len(event['Records'])} SQS record(s)")

    failed_message_ids = []

    for record in event["Records"]:
        message_id = record["messageId"]
        try:
            body = json.loads(record["body"])
            job_id = body["jobId"]
            target_flavors = body["flavors"]
            mandatory_ingredients = body.get("mandatory_ingredients", [])
            mandatory_percentages = body.get("mandatory_percentages", [])

            logger.info(f"Processing jobId={job_id} | flavors={target_flavors}")

            updateJobStatus(job_id, "RUNNING", {
                "startedAt": int(time.time()),
                "flavors": target_flavors,
                "mandatory_ingredients": mandatory_ingredients
            })

            t0 = time.time()
            best, log = run_ga(target_flavors, mandatory_ingredients, mandatory_percentages)
            elapsed = round(time.time() - t0, 2)
            recipe = formatRecipe(best)
            fitness = float(best.fitness.values[0])

            updateJobStatus(
                job_id,
                "COMPLETE",
                {
                    "recipe": recipe,
                    "fitness_score": fitness,
                    "execution_seconds": elapsed,
                    "generations_run": GA_CONFIG["GENERATIONS"],
                    "completedAt": int(time.time()),
                }
            )

            cloudwatch.put_metric_data(
                Namespace="CocktailAI/GA",
                MetricData=[
                    {
                        "MetricName": "ExecutionTime",
                        "Value": float(elapsed),
                        "Unit": "Seconds",
                        "Dimensions": [{"Name": "EndpointName", "Value": ENDPOINT_NAME}],
                    },
                    {
                        "MetricName": "FinalFitnessScore",
                        "Value": float(fitness),
                        "Unit": "None",
                        "Dimensions": [{"Name": "EndpointName", "Value": ENDPOINT_NAME}],
                    }
                ]
            )

            logger.info(f"jobId={job_id} COMPLETE | fitness={fitness:.4f} | elapsed={elapsed}s | ingredients={len(recipe)}")

        except Exception as exc:
            logger.exception(f"Failed processing messageId={message_id}: {exc}")
            try:
                job_id = json.loads(record["body"]).get("jobId", "unknown")
                updateJobStatus(job_id, "FAILED", {"error": str(exc), "failedAt": int(time.time())})
            except Exception:
                pass
            failed_message_ids.append(message_id)

    if failed_message_ids:
        return {
            "batchItemFailures": [
                {"itemIdentifier": mid} for mid in failed_message_ids
            ]
        }

    return {"statusCode": 200}