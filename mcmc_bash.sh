

python run_mcmc.py real_data/Exp_1_live_lowpH.csv \
  --alpha 0.05 \
  --mu 0.38:0.42 \
  --omega 0 \
  --rho 1 \
  --kappa-samples-path kappa_samples_stratified_highpH.npz \
  --outfile mcmc_res.csv

python run_mcmc.py real_data/Exp_2_reduced_live_lowpH.csv \
  --alpha 0.05 \
  --mu 0.20:0.28 \
  --omega 0 \
  --rho 0.08:0.20 \
  --t-switch 0.0 \
  --kappa-samples-path kappa_samples_stratified_highpH.npz \
  --outfile mcmc_res.csv

python run_mcmc.py real_data/Exp_3_time-limited_lowpH.csv \
  --alpha 0.05 \
  --mu 0.38:0.50 \
  --omega 0 \
  --rho 0.06:0.22 \
  --t-switch 24.0 \
  --kappa-samples-path kappa_samples_stratified_highpH.npz \
  --outfile mcmc_res.csv


python run_mcmc.py real_data/Exp_4_live_highpH.csv \
  --alpha 0.05 \
  --mu 0.38:0.42 \
  --omega 0 \
  --rho 1 \
  --kappa-samples-path kappa_samples_stratified_highpH.npz \
  --outfile mcmc_res.csv

  python run_mcmc.py real_data/Exp_5_time-limited_highpH.csv \
  --alpha 0.05 \
  --mu 0.30:0.40 \
  --omega 0 \
  --rho 0.08:0.2 \
  --t-switch 24.0 \
  --kappa-samples-path kappa_samples_stratified_highpH.npz \
  --outfile mcmc_res.csv


  python run_mcmc.py real_data/Exp_6_BB_time-limited_highpH.csv \
  --alpha 0.05 \
  --mu 0.22:0.36 \
  --omega 0 \
  --rho 0.02:0.14 \
  --t-switch 24.0 \
  --kappa-samples-path kappa_samples_stratified_highpH.npz \
  --outfile mcmc_res.csv


  # python run_mcmc.py real_data/Exp_7_low-salinity_time-limited.csv \
  # --alpha 0.04 \
  # --mu 0.34:0.44 \
  # --omega 0 \
  # --rho 0.14:0.38 \
  # --t-switch 24.0 \
  # --kappa-samples-path kappa_samples_stratified_highpH.npz \
  # --outfile mcmc_res.csv

  # python run_mcmc.py real_data/Exp_8_BB_low-salinity_time-limited.csv \
  # --alpha 0.04 \
  # --mu 0.28:0.44 \
  # --omega 0 \
  # --rho 0.02:0.14 \
  # --t-switch 24.0 \
  # --kappa-samples-path kappa_samples_stratified_highpH.npz \
  # --outfile mcmc_res.csv