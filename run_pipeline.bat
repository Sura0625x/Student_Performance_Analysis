@echo off
title Student Performance System - Pipeline Runner
cls
echo ===============================================================================
echo   STUDENT PERFORMANCE SYSTEM - MASTER MACHINE LEARNING PIPELINE
echo ===============================================================================
echo.
echo Running full data mining pipeline:
echo - Data Preprocessing & Cleaning
echo - 7 Exploratory Data Analysis Plots (outputs/graphs/)
echo - K-Means Clustering & Dynamic Cohort Profiling
echo - Decision Tree Training & Model Serialization (models/trained_model.pkl)
echo - Holdout Model Evaluation & Report Generation (outputs/reports/)
echo.

python run_all_pipeline.py

echo.
pause
