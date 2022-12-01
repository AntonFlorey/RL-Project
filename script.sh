#!/bin/bash  
git pull origin main
git add .
git commit -m "$*"
git push origin main