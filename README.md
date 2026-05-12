# CocktailAI — Evolutionary Cocktail Recipe Generation on AWS

AI-powered cocktail recipe generator that combines **genetic algorithms** with **ML taste classifiers** and a **serverless AWS architecture** to produce novel, user-guided recipes at scale.

## Why this project stands out
- **Hybrid AI system**: Genetic Algorithm (GA) search guided by **9 Logistic Regression taste classifiers**.
- **Serverless, asynchronous architecture**: Vercel + API + SQS + Lambda + SageMaker + DynamoDB.
- **Cost- and performance-optimized inference**: Batch GA population scoring in a **single SageMaker endpoint call** per generation.
- **Production-grade observability**: CloudWatch custom metrics + AWS X-Ray tracing.
- **Real, messy data engineering**: 4,500 scraped recipes → 2,700 labeled → **242 engineered ingredient features**.

---

## Product overview
CocktailAI generates recipes from **user-selected flavor profiles** (e.g., citrus, sweet, herbal) and optional **mandatory ingredients** (e.g., gin). The system searches a large ingredient space to output a recipe that aligns with the target tastes and practical constraints.

### End-to-end flow
1. **User submits flavor preferences** via UI.
2. **Job created in DynamoDB** with status `QUEUED`.
3. **SQS triggers Lambda** to run GA optimization.
4. Lambda calls **SageMaker taste predictor** for GA fitness.
5. **Best recipe stored** in DynamoDB; UI polls until `COMPLETE`.

---

## Architecture (AWS)
**Five-layer asynchronous ecosystem**:
- **Frontend + API**: Next.js (Vercel) with API routes
- **Queue**: SQS job queue
- **Compute**: Lambda (GA execution)
- **ML Inference**: SageMaker SKLearn endpoint
- **Data + Observability**: DynamoDB, S3, CloudWatch, X-Ray

Key decisions:
- **SQS over direct HTTP** to avoid timeouts (GA ~45s runtime).
- **Batch inference** for GA generations to minimize endpoint cost.
- **Custom metrics** for fitness, execution time, and performance tracking.

---

## ML + Optimization
### Taste classification
- **9 binary classifiers** (bittersweet, citrus, creamy, floral, fruity, herbal, savoury, spicy, sweet)
- **L1-regularized Logistic Regression**, class-weighted
- **Custom thresholds** per class to handle imbalance

**Classifier performance (AUC / F1 / Threshold):**
| Flavor | AUC | F1 | Threshold |
|---|---:|---:|---:|
| Creamy | 0.995 | 0.894 | 0.60 |
| Floral | 0.968 | 0.578 | 0.90 |
| Bittersweet | 0.956 | 0.800 | 0.50 |
| Sweet | 0.946 | 0.556 | 0.80 |
| Spicy | 0.944 | 0.580 | 0.75 |
| Citrus | 0.893 | 0.772 | 0.50 |
| Savoury | 0.884 | 0.457 | 0.80 |
| Fruity | 0.881 | 0.674 | 0.60 |
| Herbal | 0.821 | 0.516 | 0.55 |

### Genetic Algorithm (GA)
- **Population:** 1000 individuals
- **Generations:** 150
- **Fitness:** 80% taste alignment + 20% simplicity (fewer ingredients)
- **Constraints:** ingredient bounds, mandatory ingredients, total % normalization
- **Implementation:** DEAP framework

---

## Data engineering
- **Source:** Difford’s Guide cocktail recipes
- **Raw scraped recipes:** ~4,500
- **Usable labeled recipes:** ~2,700 with full flavor tags
- **Feature engineering:** 683 → **242 consolidated ingredient features**
- **Challenges addressed:** inconsistent units, duplicate ingredient names, sparse labels

---

## Repository map (important folders)
> **Note:** Archive folder is intentionally ignored.

- **Root** - deployment-ready AWS code
  - `ga_handler.py` - Lambda GA execution + DynamoDB updates + CloudWatch metrics
  - `sagemaker_script.py` - training + custom SageMaker inference hooks
  - `taste-classifier.ipynb` - training notebook for classifiers
- **`data processing/`** - scraping, cleaning, feature engineering steps
- **`mk2/`, `mk3/`** - model + experimentation iterations

---

## What this demonstrates (recruiter-focused)
- **Full-stack ML systems engineering**: data → model → optimization → production
- **Cloud architecture & scalability**: AWS serverless, async jobs, observability
- **Cost-aware design**: batched inference + Vercel deployment
- **Pragmatic ML**: interpretable models + thresholds tuned for real imbalance

---

## Scalability & next steps
- **Infrastructure as Code** (CloudFormation or Terraform)
- **SageMaker endpoint auto-scaling** or model hosting on autoscaled EC2
- **Private VPC networking** for security hardening

---

## Contact
**Jai Parakh** — AI/ML + Cloud Systems Engineer

If you’re hiring for ML systems, MLOps, or cloud architecture roles, this project demonstrates real-world end-to-end delivery.
