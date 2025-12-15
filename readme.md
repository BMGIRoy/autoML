ML service for training models, generating SHAP values, and performing data analysis.

## Deployment to Render

1. Create new GitHub repo and push this code
2. Go to Railway.app
3. Click "New Project" → "Deploy from GitHub repo"
4. Select your repo
5. Railway will auto-detect Python and deploy
6. Copy the deployment URL (e.g., `https://your-app.up.railway.app`)
7. Update all backend functions in Base44 to use this URL

## Endpoints

- `POST /train` - Train and save model
- `POST /shap` - Generate SHAP values from saved model
- `POST /tree-extract` - Extract tree structure from saved model
- `POST /correlation` - Calculate correlation matrix
- `POST /cluster` - Perform clustering analysis
- `GET /health` - Health check

## Environment

No environment variables needed - uses `/tmp` for model storage.
