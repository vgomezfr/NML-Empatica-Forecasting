This repo contains:
  1. Scripts used for forecasting biomarkers gathered from the Empatica IMU at CMU's NeuroMechatronics lab.
  2. Two slide decks, one from June 2026 and one from September 2026, presenting findings and next steps.

Brief descriptions of individual files:
  ARM_IMU Presentation.pptx: Presentation given on 6/9/26 describing the project and detailing initial findings and next steps
  Valerio NML wrap-up slides.pptx: Slide deck summarizing work done and next steps to avoid duplication of effort by successor(s)
  empatica_forecast_nn.py: Python script for training MLP, RNN, GRU, and LSTM models to forecast biomarkers
  joint_act_counts.qmd/joint_heart_rates.qmd/joint_hrvs.qmd: Notebooks for fitting ETS/ARIMA models to biomarkers and backtesting

Notes on code files' usage:
  1. empatica_forecast_nn.py allows flexible construction of models through command line inputs;
     view bottom of file for order and expectation of inputs
  2. All .qmd files separately handle fitting and backtesting models for one biomarker.
     Some code is shared (e.g. functions which can be generalized across biomarkers),
     but in general you must use the correct file for the biomarker you want to model.
     Streamlining the code to one file that takes a biomarker as a parameter is desirable; process is ongoing but troublesome.
     Notebooks contain a mix of newer and older code; with some work they will be cleaned up and organized.
