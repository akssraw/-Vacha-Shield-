# Free Deployment Guide

As of March 20, 2026, the most practical free options for this repo are:

1. Hugging Face Spaces (Docker) if you want the most durable free public link.
2. Render if you want the fastest deploy directly from the GitHub repo.

## Recommended Order

### 1. Hugging Face Spaces

Why this is the best free option for a link you may share over a week:
- Hugging Face documents that free `cpu-basic` Spaces sleep after 48 hours of inactivity, not after a few minutes.
- The public Space URL remains shareable, and the app restarts on the next visit.
- This repo already includes a Dockerfile that works for a Docker Space.

Steps:
1. Create a new Hugging Face Space and choose the `Docker` SDK.
2. Mirror this repo into the Space repo, including `model.pth`, `model_calibration.json`, and one built frontend `dist` folder.
3. Keep the root `Dockerfile` as-is.
4. In the Space settings, leave the hardware on the free CPU tier unless you need faster cold starts.
5. After the build completes, open `/health` on the Space URL to confirm the model loaded.

Suggested share link:
- `https://<your-space-name>.hf.space`

### 2. Render

Why use it:
- Render connects straight to GitHub and the repo already has `render.yaml`.
- This is the easiest path if you want a public URL quickly.

Important caveat:
- Render free web services spin down after about 15 minutes of inactivity and cold-start on the next request.
- The URL still works after a week, but the first request after idle time may take a bit longer.

Steps:
1. Push this repo to GitHub with the runtime files included.
2. In Render, create a new Web Service from the repo.
3. Let Render detect `render.yaml`.
4. Deploy and then open `/health` on the Render URL.

Suggested share link:
- `https://<your-service-name>.onrender.com`

## Not Recommended For A Lasting Free Demo

### Railway

As of March 20, 2026, Railway's official pricing docs describe the free plan as experimentation with only $1 of free resources. That makes it less reliable than Hugging Face Spaces or Render for a public link you want to keep sharing.

## Files Already Prepared In This Repo

- `Dockerfile` copies only the runtime files needed for the web app.
- `.dockerignore` excludes large local datasets, Android files, and frontend source caches from deploy builds.
- `render.yaml` keeps the health check on `/health`.
- `app.py` can serve a bundled frontend from `/app/lovable-dist` during container deploys.
