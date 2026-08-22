
# MLOPs-Production-Ready-Deep-Learning-Project




## Workflows

1. Update config.yaml
2. Update params.yaml
3. Update the entity
4. Update the configuration manager in src config
5. Update the components
6. Update the pipeline 
7. Update the main.py
8. Update the dvc.yaml 

## How to run?

```bash
conda create -n chest python=3.8 -y
```

```bash
conda activate chest
```

```bash
pip install -r requirements.txt
```

```bash
python app.py
```

## Git commands

```bash
git add .

git commit -m "Updated"

git push origin main
```


### DVC cmd

1. dvc init
2. dvc repro
3. dvc dag


## AWS CI/CD Deployment with GitHub Actions

### 1. Create an IAM user for deployment

Create an IAM user with these policies:
- `AmazonEC2ContainerRegistryFullAccess`
- `AmazonEC2FullAccess`

Save the generated Access Key ID and Secret Access Key.

**Deployment flow this sets up:**
1. Build a Docker image of the source code
2. Push the image to ECR
3. Launch an EC2 instance
4. Pull the image from ECR onto EC2
5. Run the Docker image container on EC2

### 2. Create an ECR repository

Create an ECR repo to store/pull the Docker image, e.g.:
```
343218195178.dkr.ecr.us-east-1.amazonaws.com/cnnclassifier
```

### 3. Launch an EC2 instance (Ubuntu)

### 4. Install Docker on the EC2 instance

```bash
sudo apt-get update -y
sudo apt-get upgrade -y

curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
newgrp docker
```

### 5. Configure the EC2 instance as a self-hosted GitHub Actions runner

In your GitHub repo: **Settings > Actions > Runners > New self-hosted runner**, choose Linux, and run the commands it gives you on the EC2 instance.

### 6. Set up GitHub repository secrets

| Secret name | Value |
|---|---|
| `AWS_ACCESS_KEY_ID` | from the IAM user |
| `AWS_SECRET_ACCESS_KEY` | from the IAM user |
| `AWS_REGION` | e.g. `us-east-1` |
| `ECR_REPOSITORY_NAME` | e.g. `cnnclassifier` (repo name only, not the full URI) |

### 7. Push to `main`

Pushing to `main` triggers `.github/workflows/main.yaml`, which builds the Docker image, pushes it to ECR, then (via the self-hosted runner on your EC2 instance) pulls and runs it on port `8080`.

Make sure the EC2 instance's security group allows inbound traffic on port `8080`. The app will then be reachable at `http://<EC2-public-ip>:8080`.

**Note:** the `/train` route inside the app calls `dvc repro`, which needs outbound internet access (to download data and reach DagsHub for MLflow tracking) plus DagsHub auth configured on the instance. This isn't required for `/predict` (the main serving path) to work.
## Free Deployment: Hugging Face Spaces

This repo is also set up to deploy for free on [Hugging Face Spaces](https://huggingface.co/spaces) (Docker SDK, free CPU tier: 2 vCPU / 16GB RAM, no credit card required).

The `hf-space` branch is a squashed, ready-to-push snapshot of this repo (model weights tracked with Git LFS, and a README with the Spaces metadata header) so it can be pushed directly as a new Space.

### Steps

1. Create a free account at https://huggingface.co and create a new Space:
   - SDK: **Docker**
   - Visibility: your choice (public/private)
2. Add the Space as a git remote and push the `hf-space` branch to it:
   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/<space-name>
   git push space hf-space:main
   ```
   You'll be prompted for your Hugging Face username and an [access token](https://huggingface.co/settings/tokens) (use it in place of a password).
3. The Space will build the Dockerfile and start serving on port 8080 automatically. Your app will be live at:
   ```
   https://huggingface.co/spaces/<your-username>/<space-name>
   ```

To update the Space later, rebuild the `hf-space` branch from `main` (or just push new changes into it) and `git push space hf-space:main` again.
